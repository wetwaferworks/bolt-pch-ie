/* JSON parser for safetensors headers (C17). Written to the same grammar, ceilings and refusal rules as
 * MiniJson (RFC 8259 numbers, \uXXXX with surrogate pairs, no locale, duplicate keys optional). */
#include "pchie_safetensors_json.h"
#include "pchie_ternary_convert_internal.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <locale.h>

#if defined(_WIN32)
#include <locale.h>
#endif

enum {
    TERNARY_JSON_MAX_DEPTH = 64,
    TERNARY_JSON_MAX_BYTES = 16 << 20,
    TERNARY_JSON_MAX_STRING_BYTES = 16 << 20,
    TERNARY_JSON_MAX_ITEMS = 1 << 20,
    TERNARY_JSON_MAX_VALUES = 4 << 20,
    TERNARY_JSON_MAX_MEMORY = 64 << 20
};

typedef struct ternary_json_parse_state {
    const char *text;
    size_t length;
    size_t index;
    int max_depth;
    size_t max_string_bytes;
    size_t max_items;
    size_t max_values;
    size_t max_memory;
    size_t values_seen;
    size_t memory_used;
    int reject_duplicate_keys;
    char *error;
    size_t error_capacity;
} ternary_json_parse_state;

void ternary_json_value_clear(ternary_json_value *value) {
    size_t i;
    if (!value) return;
    if (value->type == TERNARY_JSON_STRING) {
        free(value->string_bytes);
    } else if (value->type == TERNARY_JSON_ARRAY) {
        for (i = 0; i < value->element_count; ++i) ternary_json_value_clear(&value->elements[i]);
        free(value->elements);
    } else if (value->type == TERNARY_JSON_OBJECT) {
        for (i = 0; i < value->member_count; ++i) {
            free(value->members[i].key);
            ternary_json_value_clear(value->members[i].value);
            free(value->members[i].value);
        }
        free(value->members);
    }
    memset(value, 0, sizeof(*value));
}

static int json_fail(ternary_json_parse_state *state, const char *message) {
    ternary_set_error(state->error, state->error_capacity, message);
    return 0;
}

static int json_charge(ternary_json_parse_state *state, size_t bytes) {
    if (bytes > state->max_memory - state->memory_used) return json_fail(state, "json: memory limit");
    state->memory_used += bytes;
    return 1;
}

static void *json_alloc(ternary_json_parse_state *state, size_t bytes) {
    void *p;
    if (!json_charge(state, bytes)) return NULL;
    p = calloc(1, bytes ? bytes : 1);
    if (!p) {
        json_fail(state, "json: out of memory");
        return NULL;
    }
    return p;
}

static void skip_ws(ternary_json_parse_state *state) {
    while (state->index < state->length) {
        const char c = state->text[state->index];
        if (c != ' ' && c != '\t' && c != '\n' && c != '\r') break;
        state->index++;
    }
}

static int lit(ternary_json_parse_state *state, const char *word) {
    size_t k = 0;
    while (word[k]) {
        if (state->index + k >= state->length || state->text[state->index + k] != word[k]) return 0;
        k++;
    }
    state->index += k;
    return 1;
}

static int json_grow(void **ptr, size_t *cap, size_t item_size, size_t needed, ternary_json_parse_state *state) {
    size_t new_cap;
    void *grown;
    if (needed <= *cap) return 1;
    new_cap = *cap ? *cap * 2 : 4;
    while (new_cap < needed) {
        if (new_cap > (SIZE_MAX / 2)) return json_fail(state, "json: memory limit");
        new_cap *= 2;
    }
    if (!json_charge(state, (new_cap - *cap) * item_size)) return 0;
    grown = realloc(*ptr, new_cap * item_size);
    if (!grown) return json_fail(state, "json: out of memory");
    if (new_cap > *cap)
        memset((char *)grown + *cap * item_size, 0, (new_cap - *cap) * item_size);
    *ptr = grown;
    *cap = new_cap;
    return 1;
}

static int append_byte(char **buf, size_t *len, size_t *cap, unsigned char byte, ternary_json_parse_state *state,
                       size_t max_len) {
    if (*len >= max_len) return json_fail(state, "json: string too long");
    if (!json_grow((void **)buf, cap, 1, *len + 1, state)) return 0;
    (*buf)[(*len)++] = (char)byte;
    return 1;
}

static int put_utf8(char **buf, size_t *len, size_t *cap, uint32_t codepoint, ternary_json_parse_state *state,
                    size_t max_len) {
    unsigned char bytes[4];
    int n = 0, i;
    if (codepoint < 0x80) {
        bytes[0] = (unsigned char)codepoint;
        n = 1;
    } else if (codepoint < 0x800) {
        bytes[0] = (unsigned char)(0xC0 | (codepoint >> 6));
        bytes[1] = (unsigned char)(0x80 | (codepoint & 0x3F));
        n = 2;
    } else if (codepoint < 0x10000) {
        bytes[0] = (unsigned char)(0xE0 | (codepoint >> 12));
        bytes[1] = (unsigned char)(0x80 | ((codepoint >> 6) & 0x3F));
        bytes[2] = (unsigned char)(0x80 | (codepoint & 0x3F));
        n = 3;
    } else {
        bytes[0] = (unsigned char)(0xF0 | (codepoint >> 18));
        bytes[1] = (unsigned char)(0x80 | ((codepoint >> 12) & 0x3F));
        bytes[2] = (unsigned char)(0x80 | ((codepoint >> 6) & 0x3F));
        bytes[3] = (unsigned char)(0x80 | (codepoint & 0x3F));
        n = 4;
    }
    for (i = 0; i < n; ++i)
        if (!append_byte(buf, len, cap, bytes[i], state, max_len)) return 0;
    return 1;
}

static int hex4(ternary_json_parse_state *state, uint32_t *out) {
    int k;
    uint32_t value = 0;
    if (state->index + 4 > state->length) return 0;
    for (k = 0; k < 4; ++k) {
        const char c = state->text[state->index + (size_t)k];
        value <<= 4;
        if (c >= '0' && c <= '9') value |= (uint32_t)(c - '0');
        else if (c >= 'a' && c <= 'f') value |= (uint32_t)(c - 'a' + 10);
        else if (c >= 'A' && c <= 'F') value |= (uint32_t)(c - 'A' + 10);
        else return 0;
    }
    state->index += 4;
    *out = value;
    return 1;
}

static int parse_string_into(ternary_json_parse_state *state, char **out_bytes, size_t *out_length) {
    char *buf = NULL;
    size_t len = 0, cap = 0;
    *out_bytes = NULL;
    *out_length = 0;
    if (state->index >= state->length || state->text[state->index] != '"')
        return json_fail(state, "json: expected string");
    state->index++;
    while (state->index < state->length) {
        unsigned char c = (unsigned char)state->text[state->index++];
        if (len > state->max_string_bytes) {
            free(buf);
            return json_fail(state, "json: string too long");
        }
        if (c == '"') {
            if (len > state->max_string_bytes) {
                free(buf);
                return json_fail(state, "json: string too long");
            }
            if (!json_grow((void **)&buf, &cap, 1, len + 1, state)) {
                free(buf);
                return 0;
            }
            buf[len] = 0;
            *out_bytes = buf;
            *out_length = len;
            return 1;
        }
        if (c < 0x20) {
            free(buf);
            return json_fail(state, "json: raw control character in string");
        }
        if (c != '\\') {
            if (!append_byte(&buf, &len, &cap, c, state, state->max_string_bytes)) {
                free(buf);
                return 0;
            }
            continue;
        }
        if (state->index >= state->length) break;
        c = (unsigned char)state->text[state->index++];
        switch (c) {
            case '"':
            case '\\':
            case '/':
                if (!append_byte(&buf, &len, &cap, c, state, state->max_string_bytes)) { free(buf); return 0; }
                break;
            case 'b':
                if (!append_byte(&buf, &len, &cap, '\b', state, state->max_string_bytes)) { free(buf); return 0; }
                break;
            case 'f':
                if (!append_byte(&buf, &len, &cap, '\f', state, state->max_string_bytes)) { free(buf); return 0; }
                break;
            case 'n':
                if (!append_byte(&buf, &len, &cap, '\n', state, state->max_string_bytes)) { free(buf); return 0; }
                break;
            case 'r':
                if (!append_byte(&buf, &len, &cap, '\r', state, state->max_string_bytes)) { free(buf); return 0; }
                break;
            case 't':
                if (!append_byte(&buf, &len, &cap, '\t', state, state->max_string_bytes)) { free(buf); return 0; }
                break;
            case 'u': {
                uint32_t codepoint = 0;
                if (!hex4(state, &codepoint)) {
                    free(buf);
                    return json_fail(state, "json: bad \\u escape");
                }
                if (codepoint >= 0xDC00 && codepoint < 0xE000) {
                    free(buf);
                    return json_fail(state, "json: unpaired low surrogate");
                }
                if (codepoint >= 0xD800 && codepoint < 0xDC00) {
                    int paired = 0;
                    if (state->index + 6 <= state->length && state->text[state->index] == '\\' &&
                        state->text[state->index + 1] == 'u') {
                        size_t saved = state->index;
                        uint32_t low = 0;
                        state->index += 2;
                        if (hex4(state, &low) && low >= 0xDC00 && low < 0xE000) {
                            codepoint = 0x10000 + ((codepoint - 0xD800) << 10) + (low - 0xDC00);
                            paired = 1;
                        } else {
                            state->index = saved;
                        }
                    }
                    if (!paired) {
                        free(buf);
                        return json_fail(state, "json: unpaired high surrogate");
                    }
                }
                if (!put_utf8(&buf, &len, &cap, codepoint, state, state->max_string_bytes)) {
                    free(buf);
                    return 0;
                }
                break;
            }
            default:
                free(buf);
                return json_fail(state, "json: bad escape");
        }
    }
    free(buf);
    return json_fail(state, "json: unterminated string");
}

static int parse_number(ternary_json_parse_state *state, double *out) {
    size_t start = state->index, end, k, digits;
    const char *num;
    size_t num_len;
    int good = 1;
    char stack[128];
    char *heap = NULL;
    char *tmp;
    char *endptr = NULL;
    double value;
    if (state->text[state->index] == '-') state->index++;
    while (state->index < state->length) {
        const char c = state->text[state->index];
        if ((c >= '0' && c <= '9') || c == '.' || c == 'e' || c == 'E' || c == '+' || c == '-')
            state->index++;
        else
            break;
    }
    end = state->index;
    num = state->text + start;
    num_len = end - start;
    k = 0;
    if (k < num_len && num[k] == '-') k++;
    digits = 0;
    if (k < num_len && num[k] == '0') {
        k++;
    } else {
        while (k < num_len && num[k] >= '0' && num[k] <= '9') {
            k++;
            digits++;
        }
        good = digits > 0;
    }
    if (good && k < num_len && num[k] == '.') {
        k++;
        digits = 0;
        while (k < num_len && num[k] >= '0' && num[k] <= '9') {
            k++;
            digits++;
        }
        good = digits > 0;
    }
    if (good && k < num_len && (num[k] == 'e' || num[k] == 'E')) {
        k++;
        if (k < num_len && (num[k] == '+' || num[k] == '-')) k++;
        digits = 0;
        while (k < num_len && num[k] >= '0' && num[k] <= '9') {
            k++;
            digits++;
        }
        good = digits > 0;
    }
    good = good && k == num_len && num_len > 0;
    if (!good) return json_fail(state, "json: bad number");
    if (num_len + 1 <= sizeof(stack)) {
        tmp = stack;
    } else {
        heap = (char *)malloc(num_len + 1);
        if (!heap) return json_fail(state, "json: out of memory");
        tmp = heap;
    }
    memcpy(tmp, num, num_len);
    tmp[num_len] = 0;
#if defined(_WIN32)
    {
        _locale_t loc = _create_locale(LC_NUMERIC, "C");
        if (!loc) {
            free(heap);
            return json_fail(state, "json: bad number");
        }
        value = _strtod_l(tmp, &endptr, loc);
        _free_locale(loc);
    }
#else
    {
        locale_t loc = newlocale(LC_NUMERIC_MASK, "C", (locale_t)0);
        if (!loc) {
            free(heap);
            return json_fail(state, "json: bad number");
        }
        value = strtod_l(tmp, &endptr, loc);
        freelocale(loc);
    }
#endif
    free(heap);
    if (!endptr || endptr != tmp + num_len || !isfinite(value)) return json_fail(state, "json: bad number");
    *out = value;
    return 1;
}

static int compare_members(const void *a, const void *b) {
    const ternary_json_member *ma = (const ternary_json_member *)a;
    const ternary_json_member *mb = (const ternary_json_member *)b;
    const size_t n = ma->key_length < mb->key_length ? ma->key_length : mb->key_length;
    int c = memcmp(ma->key, mb->key, n);
    if (c != 0) return c;
    if (ma->key_length < mb->key_length) return -1;
    if (ma->key_length > mb->key_length) return 1;
    return 0;
}

static int has_duplicate_keys(ternary_json_value *object) {
    ternary_json_member *copy;
    size_t i;
    if (object->member_count < 2) return 0;
    copy = (ternary_json_member *)malloc(object->member_count * sizeof(*copy));
    if (!copy) return 1;
    memcpy(copy, object->members, object->member_count * sizeof(*copy));
    qsort(copy, object->member_count, sizeof(*copy), compare_members);
    for (i = 1; i < object->member_count; ++i) {
        if (copy[i].key_length == copy[i - 1].key_length &&
            memcmp(copy[i].key, copy[i - 1].key, copy[i].key_length) == 0) {
            free(copy);
            return 1;
        }
    }
    free(copy);
    return 0;
}

static int parse_value(ternary_json_parse_state *state, ternary_json_value *value, int depth);

static int parse_object(ternary_json_parse_state *state, ternary_json_value *value, int depth) {
    size_t cap = 0;
    value->type = TERNARY_JSON_OBJECT;
    state->index++;
    skip_ws(state);
    if (state->index < state->length && state->text[state->index] == '}') {
        state->index++;
        return 1;
    }
    for (;;) {
        ternary_json_member *member;
        skip_ws(state);
        if (value->member_count >= state->max_items) return json_fail(state, "json: too many keys in one object");
        if (!json_grow((void **)&value->members, &cap, sizeof(*value->members), value->member_count + 1, state))
            return 0;
        member = &value->members[value->member_count];
        memset(member, 0, sizeof(*member));
        value->member_count++;
        if (!parse_string_into(state, &member->key, &member->key_length)) return 0;
        skip_ws(state);
        if (state->index >= state->length || state->text[state->index] != ':')
            return json_fail(state, "json: expected ':'");
        state->index++;
        member->value = (ternary_json_value *)json_alloc(state, sizeof(*member->value));
        if (!member->value) return 0;
        if (!parse_value(state, member->value, depth + 1)) return 0;
        skip_ws(state);
        if (state->index < state->length && state->text[state->index] == ',') {
            state->index++;
            continue;
        }
        if (state->index < state->length && state->text[state->index] == '}') {
            state->index++;
            if (state->reject_duplicate_keys && has_duplicate_keys(value))
                return json_fail(state, "json: duplicate key");
            return 1;
        }
        return json_fail(state, "json: expected ',' or '}'");
    }
}

static int parse_array(ternary_json_parse_state *state, ternary_json_value *value, int depth) {
    size_t cap = 0;
    value->type = TERNARY_JSON_ARRAY;
    state->index++;
    skip_ws(state);
    if (state->index < state->length && state->text[state->index] == ']') {
        state->index++;
        return 1;
    }
    for (;;) {
        if (value->element_count >= state->max_items) return json_fail(state, "json: too many items in one array");
        if (!json_grow((void **)&value->elements, &cap, sizeof(*value->elements), value->element_count + 1, state))
            return 0;
        memset(&value->elements[value->element_count], 0, sizeof(*value->elements));
        value->element_count++;
        if (!parse_value(state, &value->elements[value->element_count - 1], depth + 1)) return 0;
        skip_ws(state);
        if (state->index < state->length && state->text[state->index] == ',') {
            state->index++;
            continue;
        }
        if (state->index < state->length && state->text[state->index] == ']') {
            state->index++;
            return 1;
        }
        return json_fail(state, "json: expected ',' or ']'");
    }
}

static int parse_value(ternary_json_parse_state *state, ternary_json_value *value, int depth) {
    char c;
    if (depth > state->max_depth) return json_fail(state, "json: too deep");
    if (++state->values_seen > state->max_values) return json_fail(state, "json: too many values");
    skip_ws(state);
    if (state->index >= state->length) return json_fail(state, "json: unexpected end");
    memset(value, 0, sizeof(*value));
    c = state->text[state->index];
    if (c == '{') return parse_object(state, value, depth);
    if (c == '[') return parse_array(state, value, depth);
    if (c == '"') {
        value->type = TERNARY_JSON_STRING;
        return parse_string_into(state, &value->string_bytes, &value->string_length);
    }
    if (lit(state, "true")) {
        value->type = TERNARY_JSON_BOOL;
        value->boolean = 1;
        return 1;
    }
    if (lit(state, "false")) {
        value->type = TERNARY_JSON_BOOL;
        value->boolean = 0;
        return 1;
    }
    if (lit(state, "null")) {
        value->type = TERNARY_JSON_NULL;
        return 1;
    }
    if (c == '-' || (c >= '0' && c <= '9')) {
        value->type = TERNARY_JSON_NUMBER;
        return parse_number(state, &value->number);
    }
    return json_fail(state, "json: unexpected character");
}

int ternary_json_parse(const char *text, size_t length, int max_depth, size_t max_bytes, int reject_duplicate_keys,
                       ternary_json_value *out, char *error, size_t error_capacity) {
    ternary_json_parse_state state;
    memset(&state, 0, sizeof(state));
    memset(out, 0, sizeof(*out));
    if (max_bytes == 0 || max_bytes > TERNARY_JSON_MAX_BYTES) max_bytes = TERNARY_JSON_MAX_BYTES;
    if (max_depth <= 0 || max_depth > TERNARY_JSON_MAX_DEPTH) max_depth = TERNARY_JSON_MAX_DEPTH;
    if (length > max_bytes) {
        ternary_set_error(error, error_capacity, "json: input too large");
        return 0;
    }
    if (!text && length) {
        ternary_set_error(error, error_capacity, "json: unexpected end");
        return 0;
    }
    state.text = text ? text : "";
    state.length = length;
    state.max_depth = max_depth;
    state.max_string_bytes = TERNARY_JSON_MAX_STRING_BYTES;
    state.max_items = TERNARY_JSON_MAX_ITEMS;
    state.max_values = TERNARY_JSON_MAX_VALUES;
    state.max_memory = TERNARY_JSON_MAX_MEMORY;
    state.reject_duplicate_keys = reject_duplicate_keys;
    state.error = error;
    state.error_capacity = error_capacity;
    if (!parse_value(&state, out, 0)) {
        ternary_json_value_clear(out);
        return 0;
    }
    skip_ws(&state);
    if (state.index != state.length) {
        ternary_json_value_clear(out);
        ternary_set_error(error, error_capacity, "json: trailing characters");
        return 0;
    }
    return 1;
}

const ternary_json_value *ternary_json_object_get(const ternary_json_value *object, const char *key) {
    size_t i, n;
    if (!object || object->type != TERNARY_JSON_OBJECT || !key) return NULL;
    n = strlen(key);
    for (i = 0; i < object->member_count; ++i)
        if (object->members[i].key_length == n && memcmp(object->members[i].key, key, n) == 0)
            return object->members[i].value;
    return NULL;
}
