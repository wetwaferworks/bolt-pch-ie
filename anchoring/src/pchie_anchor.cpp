// pchie_anchor: local append-only anchor log, signed tree heads, proofs, file-based witness exchange, and the verifier.
// Side tool: reads finished-run artifacts; never linked into, called by, or fed back into a run. No network.
#include "pchie_anchor.h"
#include "pchie_anchor_fs.h"
#include <ctime>
#include <filesystem>
#include <iostream>
#include <algorithm>
#include <map>
using namespace pchie::anchor; namespace fs = std::filesystem;

static std::string slurp(const fs::path& p) { std::ifstream f(p, std::ios::binary); if (!f) throw std::runtime_error("cannot read " + p.string()); std::stringstream s; s << f.rdbuf(); return s.str(); }
static std::string trim(std::string s) { while (!s.empty() && (s.back() == '\n' || s.back() == '\r' || s.back() == ' ')) s.pop_back(); return s; }
// Every state file is replaced atomically through a unique, flushed temp file (see pchie_anchor_fs.cpp).
static void spit(const fs::path& p, const std::string& s) { if (p.has_parent_path()) fs::create_directories(p.parent_path()); durable_write_file(p, s); }
// Witness ids name files (cosigs/<size>_<id>.txt, witness_held/<id>.txt), so they are kept to a small safe set.
static bool witness_id_ok(const std::string& id) {
  if (id.empty() || id.size() > 64 || id[0] == '.' || id[0] == '-') return false;
  for (char c : id) if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '-' || c == '_' || c == '.')) return false;
  return id.find("..") == std::string::npos;
}
static const std::string& require_witness_id(const std::string& id) {
  if (!witness_id_ok(id)) throw std::runtime_error("bad witness id '" + id + "' (letters, digits, '-', '_', '.'; at most 64; no leading '.' or '-')");
  return id;
}
static std::string now_utc() { std::time_t t = std::time(nullptr); std::tm g{};
#ifdef _WIN32
  gmtime_s(&g, &t);
#else
  gmtime_r(&t, &g);
#endif
  char b[32]; std::strftime(b, sizeof b, "%Y-%m-%dT%H:%M:%SZ", &g); return b; }
static std::vector<std::string> lines(const fs::path& p) { std::vector<std::string> v; if (!fs::exists(p)) return v; std::ifstream f(p, std::ios::binary); std::string l; while (std::getline(f, l)) { if (!l.empty() && l.back() == '\r') l.pop_back(); if (!l.empty()) v.push_back(l); } return v; }
static std::map<std::string, std::string> kvfile(const fs::path& p) { std::map<std::string, std::string> m; for (auto& l : lines(p)) { if (l[0] == '#') continue; auto e = l.find('='); if (e != std::string::npos) m[l.substr(0, e)] = l.substr(e + 1); } return m; }

// ---- key files: secret = I, seed, next_q (persisted before each signature is released) ----
static HbsKey load_key(const fs::path& p) {
  auto m = kvfile(p); H32 seed = unhex32(m.at("seed")); HbsKey k = HbsKey::from_seed(unhex(m.at("I")), seed, uint32_t(std::stoul(m.at("next_q"))));
  secure_zero(seed.data(), seed.size()); for (auto& kv : m) wipe_string(kv.second); return k;
}
static void save_key(const fs::path& p, const HbsKey& k) {
  std::string text = "# SECRET scratch signing key (PCHIE-HBS, RFC8554-modeled). Keep it on the machine that holds the log.\nI=" + hex(k.I) + "\nseed=" + hex(k.seed.data(), 32) + "\nnext_q=" + std::to_string(k.next_q) + "\n";
  try { spit(p, text); } catch (...) { wipe_string(text); throw; }
  wipe_string(text);
}
// One-time signing leaves must never be reused. Under an exclusive lock that every process takes, read the
// key, durably reserve its next index (unique temp file, flushed, atomic rename), and only then sign with it.
// A crash after the reservation wastes one leaf; it never hands the same leaf out twice.
static Bytes sign_persist(const fs::path& keyp, const std::string& text) {
  fs::path lockp = keyp; lockp += ".lock";
  ExclusiveFileLock lock(lockp);
  HbsKey k = load_key(keyp);
  if (k.next_q >= Hbs::kLeaves) { k.wipe(); throw std::runtime_error("signing key exhausted: every one-time leaf has been used"); }
  HbsKey adv = k; adv.next_q++;
  save_key(keyp, adv); adv.wipe();
  Bytes sig = k.sign(Bytes(text.begin(), text.end())); k.wipe();
  return sig;
}
static void keygen(const fs::path& dir, const std::string& label) { HbsKey k = HbsKey::generate(label); save_key(dir / "signing.key", k); spit(dir / "public.key", hex(k.pub()) + "\n"); }

struct Log {
  fs::path d; std::vector<std::string> recs; std::vector<H32> L; std::string log_id;
  explicit Log(fs::path dir) : d(std::move(dir)) { log_id = trim(slurp(d / "log_id.txt")); recs = lines(d / "anchors.log"); for (auto& r : recs) L.push_back(leaf_hash(Bytes(r.begin(), r.end()))); }
  Bytes pub() const { return unhex(trim(slurp(d / "keys" / "public.key"))); }
  H32 root(size_t n) const { return mth(L, 0, n); }
  Sth sth(size_t n) const { return Sth::parse(trim(slurp(d / "sth" / ("sth_" + std::to_string(n) + ".txt")))); }
};
static std::string arg(int argc, char** argv, const std::string& k, const std::string& def = "") { for (int i = 1; i + 1 < argc; ++i) if (argv[i] == k) return argv[i + 1]; return def; }

static Sth make_sth(const Log& lg, size_t n) { Sth s; s.log_id = lg.log_id; s.tree_size = n; s.root = lg.root(n); s.timestamp_utc = now_utc(); s.sig = sign_persist(lg.d / "keys" / "signing.key", s.signed_text()); spit(lg.d / "sth" / ("sth_" + std::to_string(n) + ".txt"), s.line() + "\n"); spit(lg.d / "sth_latest.txt", s.line() + "\n"); return s; }

// ---- witness side (same code a remote witness would run): verify consistency from its last cosigned head, then cosign ----
struct Packet { Sth old_sth, new_sth; std::vector<H32> proof; Bytes log_pub; bool has_old = false; };
static std::string packet_text(const Packet& p) { std::string s = "pchie.witness_packet.v1\nlog_pub=" + hex(p.log_pub) + "\n"; if (p.has_old) s += "old_sth=" + p.old_sth.line() + "\n"; s += "new_sth=" + p.new_sth.line() + "\nconsistency_proof="; for (size_t i = 0; i < p.proof.size(); ++i) s += (i ? "," : "") + hex(p.proof[i]); return s + "\n"; }
static Packet parse_packet(const fs::path& f) { Packet p; auto m = kvfile(f); p.log_pub = unhex(m.at("log_pub")); if (m.count("old_sth")) { p.old_sth = Sth::parse(m["old_sth"]); p.has_old = true; } p.new_sth = Sth::parse(m.at("new_sth")); std::stringstream ss(m["consistency_proof"]); std::string h; while (std::getline(ss, h, ',')) if (!h.empty()) p.proof.push_back(unhex32(h)); return p; }
// returns "" on success, else reason
// The witness pins the log's public key: from witness-trust (configured ahead of time) or on first contact.
// Every later packet must carry that same key; a packet signed under any other key is refused.
static const char* kTrustedLogPub = "trusted_log_pub.txt";
static std::string witness_cosign(const fs::path& wdir, const std::string& wid, const Packet& p, std::string* cosig_line) {
  if (!witness_id_ok(wid)) return "bad witness id";
  fs::create_directories(wdir);
  ExclusiveFileLock lock(wdir / "witness.lock");
  const fs::path st = wdir / "last_cosigned_sth.txt", pin = wdir / kTrustedLogPub;
  const bool pinned = fs::exists(pin), held_exists = fs::exists(st);
  if (pinned) {
    if (hex(p.log_pub) != trim(slurp(pin))) return "LOG KEY CHANGED: the packet's log key is not the key this witness pinned";
  } else if (held_exists) {
    // State from before key pinning: adopt the packet's key only if it signed the head this witness already holds.
    Sth held = Sth::parse(trim(slurp(st)));
    if (!hbs_verify(p.log_pub, B(held.signed_text()), held.sig)) return "LOG KEY CHANGED: the witness-held head was not signed by the packet's log key";
  }
  if (!hbs_verify(p.log_pub, B(p.new_sth.signed_text()), p.new_sth.sig)) return "new STH signature invalid";
  if (held_exists) {
    Sth held = Sth::parse(trim(slurp(st)));
    if (held.log_id != p.new_sth.log_id) return "LOG ID CHANGED: witness holds log '" + held.log_id + "', packet is for '" + p.new_sth.log_id + "'";
    if (p.new_sth.tree_size < held.tree_size) return "ROLLBACK: new tree_size " + std::to_string(p.new_sth.tree_size) + " < witness-held " + std::to_string(held.tree_size);
    if (!verify_consistency(held.tree_size, held.root, p.new_sth.tree_size, p.new_sth.root, p.proof)) return "consistency proof from witness-held head FAILED";
  } else if (p.has_old) {
    if (!verify_consistency(p.old_sth.tree_size, p.old_sth.root, p.new_sth.tree_size, p.new_sth.root, p.proof)) return "consistency proof FAILED (first contact)";
  }
  if (!pinned) spit(pin, hex(p.log_pub) + "\n");  // pinned before the first cosignature is released
  const std::string t = cosig_text(wid, p.new_sth); Bytes sig = sign_persist(wdir / "signing.key", t);
  spit(st, p.new_sth.line() + "\n"); *cosig_line = t + "|sig=" + hex(sig); return "";
}

// Optional secret genesis salt (off by default). The file holds the salt as hex; it is read only when given.
static Bytes read_salt(const std::string& path) { if (path.empty()) return Bytes(); return unhex(trim(slurp(path))); }

// ---- witness policy: a configurable list of N witnesses plus a threshold, and optionally witnesses that
// must always sign. The defined policy is three-way (jd-local, cortana-producer, elfie); the bench policy needs jd-local (local machine) and
// Cortana, with Elfie listed but not required. Public keys come from the log's witness_registry.txt.
struct WitnessPolicy { std::string name; std::vector<std::string> witnesses, required; size_t threshold = 0; };
static std::vector<std::string> split_list(const std::string& s) { std::vector<std::string> v; std::stringstream ss(s); std::string x; while (std::getline(ss, x, ',')) { x = trim(x); while (!x.empty() && x.front() == ' ') x.erase(x.begin()); if (!x.empty()) v.push_back(x); } return v; }
static WitnessPolicy load_policy(const fs::path& f) {
  auto m = kvfile(f); WitnessPolicy P; P.name = m["policy"]; P.witnesses = split_list(m["witnesses"]); P.required = split_list(m["required"]);
  P.threshold = m.count("threshold") ? size_t(std::stoul(m["threshold"])) : 0;
  if (P.witnesses.empty() || P.threshold == 0 || P.threshold > P.witnesses.size()) throw std::runtime_error("witness policy " + f.string() + ": need witnesses and 1 <= threshold <= number of witnesses");
  for (auto& w : P.witnesses) require_witness_id(w);
  for (auto& r : P.required) if (std::find(P.witnesses.begin(), P.witnesses.end(), r) == P.witnesses.end()) throw std::runtime_error("witness policy: required witness " + r + " is not listed");
  return P;
}
static bool check_policy(const Log& lg, const Sth& s, const WitnessPolicy& P, std::string& det) {
  auto reg = kvfile(lg.d / "witness_registry.txt"); std::vector<std::string> ok, missing;
  for (auto& w : P.witnesses) {
    bool good = false; const fs::path c = lg.d / "cosigs" / (std::to_string(s.tree_size) + "_" + w + ".txt");
    if (reg.count(w) && reg[w] != "UNREGISTERED" && fs::exists(c)) { const std::string l = trim(slurp(c)); const auto p = l.rfind("|sig=");
      if (p != std::string::npos) { const std::string t = l.substr(0, p); good = t == cosig_text(w, s) && hbs_verify(unhex(reg[w]), Bytes(t.begin(), t.end()), unhex(l.substr(p + 5))); } }
    (good ? ok : missing).push_back(w);
  }
  bool req = true; std::string req_missing; for (auto& r : P.required) if (std::find(ok.begin(), ok.end(), r) == ok.end()) { req = false; req_missing += (req_missing.empty() ? "" : ",") + r; }
  auto join = [](const std::vector<std::string>& v) { std::string o; for (auto& x : v) o += (o.empty() ? "" : ",") + x; return o.empty() ? std::string("none") : o; };
  det = "policy=" + P.name + " tree_size=" + std::to_string(s.tree_size) + " valid=" + std::to_string(ok.size()) + "/" + std::to_string(P.witnesses.size()) + " [" + join(ok) + "] need " + std::to_string(P.threshold) +
        (P.required.empty() ? std::string() : " required=" + join(P.required) + (req ? " present" : " MISSING " + req_missing)) + " not signed: " + join(missing);
  return ok.size() >= P.threshold && req;
}

int pchie_anchor_main(int argc, char** argv) try {
  if (argc < 2) { std::cerr << "usage: pchie_anchor <init|emit|witness-keygen|witness-trust|cosign-local|witness-cosign|witness-packet|intake|verify|fingerprint|prove-inclusion|prove-consistency> ...\n"; return 2; }
  const std::string cmd = argv[1];
  if (cmd == "init") {  // init <dir> <log_id>
    fs::path d = argv[2]; if (fs::exists(d / "log_id.txt")) throw std::runtime_error("log exists; append-only, refusing re-init");
    fs::create_directories(d / "keys"); spit(d / "log_id.txt", std::string(argv[3]) + "\n"); keygen(d / "keys", "pchie.anchor.logkey.v1"); spit(d / "anchors.log", "");
    std::cout << "log_pub=" << trim(slurp(d / "keys" / "public.key")) << "\n"; return 0;
  }
  if (cmd == "witness-keygen") { fs::path w = argv[2]; if (fs::exists(w / "signing.key")) throw std::runtime_error("witness key exists"); fs::create_directories(w); keygen(w, "pchie.anchor.witnesskey.v1"); std::cout << "witness_pub=" << trim(slurp(w / "public.key")) << "\n"; return 0; }
  if (cmd == "witness-trust") {  // witness-trust <witness_dir> <log_pub_hex>: pin the log key before first contact
    if (argc < 4) throw std::runtime_error("usage: witness-trust <witness_dir> <log_pub_hex>");
    const fs::path w = argv[2]; const std::string want = hex(unhex(argv[3])); if (unhex(want).size() != 48) throw std::runtime_error("log_pub must be 48 bytes (96 hex digits)");
    fs::create_directories(w); ExclusiveFileLock lock(w / "witness.lock"); const fs::path pin = w / kTrustedLogPub;
    if (fs::exists(pin)) { if (trim(slurp(pin)) != want) { std::cout << "WITNESS REFUSED: a different log key is already pinned\n"; return 1; } std::cout << "log key already pinned\n"; return 0; }
    if (fs::exists(w / "last_cosigned_sth.txt")) { Sth held = Sth::parse(trim(slurp(w / "last_cosigned_sth.txt"))); if (!hbs_verify(unhex(want), B(held.signed_text()), held.sig)) { std::cout << "WITNESS REFUSED: the witness-held head was not signed by that key\n"; return 1; } }
    spit(pin, want + "\n"); std::cout << "pinned log key " << want.substr(0, 16) << "...\n"; return 0;
  }
  if (cmd == "fingerprint") { std::cout << fingerprint_for(argv[2], argv[3], argc > 4 ? argv[4] : "") << "\n"; return 0; }  // fingerprint <head> <run_id> [chain_generation]
  if (cmd == "emit") {  // emit <dir> --run-id --label --weights (--chain-file F | --chain-head H --chain-length N) [--witness-dir W --witness-id ID]
    Log lg(argv[2]); AnchorRecord a; a.run_id = arg(argc, argv, "--run-id"); a.label = arg(argc, argv, "--label"); a.weights_sha256 = arg(argc, argv, "--weights");
    if (a.run_id.empty() || a.run_id.find('|') != std::string::npos) throw std::runtime_error("bad run_id"); if (!AnchorRecord::label_ok(a.label)) throw std::runtime_error("label must be Official|Scratch|Provisional");
    unhex32(a.weights_sha256);
    const std::string cf = arg(argc, argv, "--chain-file");
    if (!cf.empty() && is_ratchet_chain_file(cf)) {  // ratcheted generation: re-derive every seal, bind to the run and weights, record generation and epoch
      const Bytes salt = read_salt(arg(argc, argv, "--genesis-salt-file"));
      RatchetFileResult rr = rederive_ratchet_chain(cf, &salt); if (!rr.ok) throw std::runtime_error("ratcheted chain file does not re-derive: " + rr.error + " (refusing to anchor)");
      if (rr.epoch_run_id != a.run_id) throw std::runtime_error("chain file belongs to run " + rr.epoch_run_id + ", not " + a.run_id + " (refusing to anchor)");
      if (rr.weights_sha256 != a.weights_sha256) throw std::runtime_error("chain file genesis weights differ from --weights (refusing to anchor)");
      a.chain_head_sha256 = hex(rr.head); a.chain_length = rr.length; a.chain_generation = rr.generation; a.epoch_id = rr.epoch_id;
      const std::string expect = arg(argc, argv, "--expect-head"); if (!expect.empty() && expect != a.chain_head_sha256) throw std::runtime_error("chain file head " + a.chain_head_sha256 + " != run-reported head " + expect + " (refusing to anchor)"); }
    else if (!cf.empty()) { H32 h; uint64_t n; if (!rederive_visnav_chain(cf, h, n)) throw std::runtime_error("cannot read chain file"); a.chain_head_sha256 = hex(h); a.chain_length = n;
      const std::string expect = arg(argc, argv, "--expect-head"); if (!expect.empty() && expect != a.chain_head_sha256) throw std::runtime_error("chain file head " + a.chain_head_sha256 + " != run-reported head " + expect + " (refusing to anchor)"); }
    else { a.chain_head_sha256 = hex(unhex32(arg(argc, argv, "--chain-head"))); a.chain_length = std::stoull(arg(argc, argv, "--chain-length")); }
    // One emit at a time per log: re-read the log under its lock, then check, link, append durably and sign the
    // new head before letting go. Without it two emits read the same tail, both link to it and fork the chain.
    ExclusiveFileLock log_lock(lg.d / "anchors.lock");
    lg = Log(lg.d);
    for (auto& r : lg.recs) if (AnchorRecord::parse(r).run_id == a.run_id) throw std::runtime_error("run_id already anchored: " + a.run_id);
    a.prev_anchor_sha256 = lg.recs.empty() ? std::string(64, '0') : hex(sha(lg.recs.back())); a.emitted_utc = arg(argc, argv, "--emitted", now_utc());
    const std::string line = a.canonical(); durable_append_file(lg.d / "anchors.log", line + "\n");
    lg.recs.push_back(line); lg.L.push_back(leaf_hash(Bytes(line.begin(), line.end())));
    Sth s = make_sth(lg, lg.L.size());
    std::cout << "anchor_index=" << lg.L.size() - 1 << "\nrecord_sha256=" << hex(a.record_hash()) << "\nchain_head_sha256=" << a.chain_head_sha256 << "\nchain_length=" << a.chain_length
              << (a.chain_generation.empty() ? std::string() : "\nchain_generation=" + a.chain_generation + "\nepoch_id=" + a.epoch_id) << "\ntree_size=" << s.tree_size << "\nroot=" << hex(s.root) << "\nfingerprint=" << fingerprint_for(a.chain_head_sha256, a.run_id, a.chain_generation) << "\n";
    const std::string wd = arg(argc, argv, "--witness-dir"), wid = arg(argc, argv, "--witness-id");
    if (!wd.empty()) {
      require_witness_id(wid);  // local producer witness (Cortana) cosigns in-process, same rule as any witness
      Packet p; p.log_pub = lg.pub(); p.new_sth = s; fs::path st = fs::path(wd) / "last_cosigned_sth.txt";
      if (fs::exists(st)) { Sth held = Sth::parse(trim(slurp(st))); p.old_sth = held; p.has_old = true; p.proof = consistency_proof(lg.L, held.tree_size, s.tree_size); }
      std::string cl; std::string why = witness_cosign(wd, wid, p, &cl); if (!why.empty()) throw std::runtime_error("local witness refused: " + why);
      spit(lg.d / "cosigs" / (std::to_string(s.tree_size) + "_" + wid + ".txt"), cl + "\n"); std::cout << "cosigned_by=" << wid << "\n";
    }
    return 0;
  }
  if (cmd == "witness-packet") {  // witness-packet <dir> <witness_id> <outfile>: tree heads + consistency proof ONLY (no records, no chains, no run data)
    Log lg(argv[2]); const std::string wid = require_witness_id(argv[3]); Packet p; p.log_pub = lg.pub(); p.new_sth = Sth::parse(trim(slurp(lg.d / "sth_latest.txt")));
    const fs::path held = lg.d / "witness_held" / (wid + ".txt");  // last head this witness cosigned (as recorded at intake)
    if (fs::exists(held)) { p.old_sth = Sth::parse(trim(slurp(held))); p.has_old = true; p.proof = consistency_proof(lg.L, p.old_sth.tree_size, p.new_sth.tree_size); }
    spit(argv[4], packet_text(p)); std::cout << "packet=" << argv[4] << " new_tree_size=" << p.new_sth.tree_size << (p.has_old ? " from=" + std::to_string(p.old_sth.tree_size) : std::string(" first-contact")) << " proof_nodes=" << p.proof.size() << "\n"; return 0;
  }
  if (cmd == "cosign-local") {  // cosign-local <dir> <witness_dir> <witness_id>: a witness on this machine cosigns the latest tree head (files only, no network)
    Log lg(argv[2]); const fs::path wd = argv[3]; const std::string wid = require_witness_id(argv[4]); Packet p; p.log_pub = lg.pub(); p.new_sth = Sth::parse(trim(slurp(lg.d / "sth_latest.txt")));
    const fs::path st = wd / "last_cosigned_sth.txt";
    if (fs::exists(st)) { Sth held = Sth::parse(trim(slurp(st))); p.old_sth = held; p.has_old = true; if (held.tree_size <= p.new_sth.tree_size) p.proof = consistency_proof(lg.L, held.tree_size, p.new_sth.tree_size); }
    std::string cl; const std::string why = witness_cosign(wd, wid, p, &cl); if (!why.empty()) { std::cout << "WITNESS REFUSED: " << why << "\n"; return 1; }
    spit(lg.d / "cosigs" / (std::to_string(p.new_sth.tree_size) + "_" + wid + ".txt"), cl + "\n"); spit(lg.d / "witness_held" / (wid + ".txt"), p.new_sth.line() + "\n");
    std::cout << "cosigned_by=" << wid << " tree_size=" << p.new_sth.tree_size << "\n"; return 0;
  }
  if (cmd == "witness-cosign") {  // witness-cosign <witness_dir> <witness_id> <packet> <out_cosig>
    Packet p = parse_packet(argv[4]); std::string cl; std::string why = witness_cosign(argv[2], require_witness_id(argv[3]), p, &cl);
    if (!why.empty()) { std::cout << "WITNESS REFUSED: " << why << "\n"; return 1; } spit(argv[5], cl + "\n"); std::cout << "cosigned tree_size=" << p.new_sth.tree_size << "\n"; return 0;
  }
  if (cmd == "intake") {  // intake <dir> <witness_id> <cosig_file>: verify a returned countersignature against the registered witness key
    Log lg(argv[2]); const std::string wid = require_witness_id(argv[3]); auto reg = kvfile(lg.d / "witness_registry.txt");
    if (!reg.count(wid) || reg[wid] == "UNREGISTERED") { std::cout << "INTAKE FAIL: witness '" << wid << "' has no registered public key (slot open)\n"; return 1; }
    const std::string l = trim(slurp(argv[4])); const auto p = l.rfind("|sig="); const std::string t = l.substr(0, p); const Bytes sig = unhex(l.substr(p + 5));
    const auto ts = t.find("|tree_size="); const uint64_t n = std::stoull(t.substr(ts + 11, t.find('|', ts + 1) - ts - 11)); Sth s = lg.sth(n);
    if (t != cosig_text(wid, s)) { std::cout << "INTAKE FAIL: cosignature does not match our STH at size " << n << "\n"; return 1; }
    if (!hbs_verify(unhex(reg[wid]), Bytes(t.begin(), t.end()), sig)) { std::cout << "INTAKE FAIL: bad signature\n"; return 1; }
    spit(lg.d / "cosigs" / (std::to_string(n) + "_" + wid + ".txt"), l + "\n"); spit(lg.d / "witness_held" / (wid + ".txt"), s.line() + "\n"); std::cout << "INTAKE PASS: " << wid << " cosigned tree_size=" << n << "\n"; return 0;
  }
  if (cmd == "prove-inclusion") { Log lg(argv[2]); size_t i = std::stoul(argv[3]), n = argc > 4 ? std::stoul(argv[4]) : lg.L.size(); for (auto& h : inclusion_proof(lg.L, i, n)) std::cout << hex(h) << "\n"; return 0; }
  if (cmd == "prove-consistency") { Log lg(argv[2]); for (auto& h : consistency_proof(lg.L, std::stoul(argv[3]), std::stoul(argv[4]))) std::cout << hex(h) << "\n"; return 0; }
  if (cmd == "verify") {  // verify <dir> --anchor-index i [--chain-file F [--rule v1|v2] [--genesis-salt-file S]] [--policy P] [--fingerprint "..."] [--witness-held STHFILE] [--witness-pub HEX --witness-id ID]
    Log lg(argv[2]); const size_t i = std::stoul(arg(argc, argv, "--anchor-index")); int fails = 0;
    auto R = [&](const std::string& name, bool ok, const std::string& det) { std::cout << (ok ? "PASS " : "FAIL ") << name << "  " << det << "\n"; if (!ok) ++fails; };
    if (i >= lg.recs.size()) { R("anchor_present", false, "index out of range"); return 1; }
    AnchorRecord a = AnchorRecord::parse(lg.recs[i]); Sth s = Sth::parse(trim(slurp(lg.d / "sth_latest.txt")));
    R("sth_signature", hbs_verify(lg.pub(), B(s.signed_text()), s.sig), "tree_size=" + std::to_string(s.tree_size) + " root=" + hex(s.root).substr(0, 16));
    R("sth_root_recompute", s.tree_size <= lg.L.size() && lg.root(s.tree_size) == s.root, "log entries reproduce STH root");
    R("inclusion", verify_inclusion(lg.L[i], i, s.tree_size, inclusion_proof(lg.L, i, s.tree_size), s.root), "anchor " + std::to_string(i) + " in tree " + std::to_string(s.tree_size));
    { bool ok = true; for (size_t j = 0; j < lg.recs.size(); ++j) { auto aj = AnchorRecord::parse(lg.recs[j]); if (aj.prev_anchor_sha256 != (j ? hex(sha(lg.recs[j - 1])) : std::string(64, '0'))) ok = false; } R("anchor_link", ok, "prev_anchor_sha256 chain over " + std::to_string(lg.recs.size()) + " records"); }
    const std::string wh = arg(argc, argv, "--witness-held");
    if (!wh.empty()) { Sth h = Sth::parse(trim(slurp(wh))); bool ok = h.tree_size <= lg.L.size() && verify_consistency(h.tree_size, h.root, s.tree_size, s.root, h.tree_size <= s.tree_size ? consistency_proof(lg.L, h.tree_size, s.tree_size) : std::vector<H32>{});
      R("consistency_vs_witness", ok, "witness-held size=" + std::to_string(h.tree_size) + " -> log size=" + std::to_string(s.tree_size)); }
    const std::string wp = arg(argc, argv, "--witness-pub"), wid = arg(argc, argv, "--witness-id");
    if (!wp.empty()) { require_witness_id(wid); }
    if (!wp.empty()) { fs::path c = lg.d / "cosigs" / (std::to_string(s.tree_size) + "_" + wid + ".txt"); bool ok = false; if (fs::exists(c)) { std::string l = trim(slurp(c)); auto p = l.rfind("|sig="); std::string t = l.substr(0, p); ok = t == cosig_text(wid, s) && hbs_verify(unhex(wp), Bytes(t.begin(), t.end()), unhex(l.substr(p + 5))); } R("witness_cosig", ok, wid + " on tree_size=" + std::to_string(s.tree_size)); }
    const std::string pol = arg(argc, argv, "--policy");
    if (!pol.empty()) { WitnessPolicy P = load_policy(pol); std::string det; R("witness_policy", check_policy(lg, s, P, det), det); }
    const std::string cf = arg(argc, argv, "--chain-file");
    std::string head = a.chain_head_sha256;
    if (!cf.empty()) {
      // The re-derivation rule is the anchor's chain generation unless --rule names another (used to show generations stay apart).
      const std::string anchored_gen = a.chain_generation.empty() ? kVisnavGenerationId : a.chain_generation;
      std::string rule = arg(argc, argv, "--rule", anchored_gen);
      if (rule == "v1") rule = kVisnavGenerationId; else if (rule == "v2") rule = pchie::is_ratchet_generation(anchored_gen) ? anchored_gen : pchie::kRatchetGenerationId;
      const std::string rf_gen = ratchet_file_generation(cf); const std::string file_gen = rf_gen.empty() ? kVisnavGenerationId : rf_gen;
      R("chain_generation", rule == anchored_gen && file_gen == anchored_gen, "anchor=" + anchored_gen + " file=" + file_gen + " rule=" + rule);
      if (pchie::is_ratchet_generation(rule)) {
        const Bytes salt = read_salt(arg(argc, argv, "--genesis-salt-file"));
        RatchetFileResult rr = rederive_ratchet_chain(cf, &salt); head = rr.ok ? hex(rr.head) : std::string(64, '0');
        R("chain_rederive", rr.ok && head == a.chain_head_sha256 && rr.length == a.chain_length,
          rr.ok ? "file head=" + head.substr(0, 16) + " len=" + std::to_string(rr.length) + " vs anchored head=" + a.chain_head_sha256.substr(0, 16) + " len=" + std::to_string(a.chain_length) : rr.error);
        if (rr.ok) {
          R("genesis_binding", rr.epoch_run_id == a.run_id && rr.weights_sha256 == a.weights_sha256, "chain run=" + rr.epoch_run_id + " (genesis run=" + rr.genesis_run_id + ") weights=" + rr.weights_sha256.substr(0, 16));
          R("epoch", rr.epoch_id == a.epoch_id, "file epoch=" + rr.epoch_id.substr(0, 18) + " anchored epoch=" + (a.epoch_id.empty() ? std::string("(none)") : a.epoch_id.substr(0, 18)));
        }
      } else if (rule == kVisnavGenerationId) {
        H32 h; uint64_t n; bool rd = rederive_visnav_chain(cf, h, n); head = hex(h);
        R("chain_rederive", rd && head == a.chain_head_sha256 && n == a.chain_length, "file head=" + head.substr(0, 16) + " len=" + std::to_string(n) + " vs anchored head=" + a.chain_head_sha256.substr(0, 16) + " len=" + std::to_string(a.chain_length));
      } else R("chain_rederive", false, "unknown rule " + rule);
    }
    const std::string fp = arg(argc, argv, "--fingerprint");
    if (!fp.empty()) { std::string got = fingerprint_for(head, a.run_id, a.chain_generation); R("fingerprint", got == fp, "presented-chain fp=" + got + (a.chain_generation.empty() ? " (v1 rule)" : " (v2 rule, covers chain_generation)")); }
    std::cout << (fails ? "OVERALL FAIL" : "OVERALL PASS") << " run_id=" << a.run_id << " label=" << a.label << (a.chain_generation.empty() ? std::string() : " generation=" + a.chain_generation) << " (anchoring never upgrades a label)\n"; return fails ? 1 : 0;
  }
  throw std::runtime_error("unknown command " + cmd);
} catch (const std::exception& e) { std::cerr << "ERROR: " << e.what() << "\n"; return 3; }

#ifndef PCHIE_ANCHOR_NO_MAIN
int main(int argc, char** argv) { return pchie_anchor_main(argc, argv); }
#endif
