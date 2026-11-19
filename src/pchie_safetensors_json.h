/* Private JSON value + parser for safetensors headers (C17). Matches MiniJson's grammar and limits. */
#ifndef PCHIE_SAFETENSORS_JSON_H
#define PCHIE_SAFETENSORS_JSON_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum ternary_json_type {
    TERNARY_JSON_NULL = 0,
    TERNARY_JSON_BOOL,
    TERNARY_JSON_NUMBER,
    TERNARY_JSON_STRING,
    TERNARY_JSON_ARRAY,
    TERNARY_JSON_OBJECT
} ternary_json_type;

typedef struct ternary_json_value ternary_json_value;

typedef struct ternary_json_member {
    char *key;
    size_t key_length;
    ternary_json_value *value;
} ternary_json_member;

struct ternary_json_value {
    ternary_json_type type;
    int boolean;
    double number;
    char *string_bytes;
    size_t string_length;
    ternary_json_value *elements;
    size_t element_count;
    ternary_json_member *members;
    size_t member_count;
};

void ternary_json_value_clear(ternary_json_value *value);

/* Parses one JSON value and requires the rest of the text to be only whitespace. On failure out is cleared. */
int ternary_json_parse(const char *text, size_t length, int max_depth, size_t max_bytes, int reject_duplicate_keys,
                       ternary_json_value *out, char *error, size_t error_capacity);

const ternary_json_value *ternary_json_object_get(const ternary_json_value *object, const char *key);

#ifdef __cplusplus
}
#endif

#endif
