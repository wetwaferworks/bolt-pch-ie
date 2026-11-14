// End-to-end checks for the anchoring side tool, one block per check in the anchoring design
// "Transparency Anchoring for the PCHIE Hash Chain". Everything runs in a temp folder under the build
// directory with throwaway keys generated here; nothing is sent anywhere. Labels stay Scratch.
#define PCHIE_ANCHOR_NO_MAIN
#include "pchie_anchor.cpp"
#include <atomic>
#include <set>
#include <thread>
#include "pchie/DeterministicJoin.h"

static int passes = 0, failures = 0;
static void check(bool ok, const std::string& name) { std::cout << "[ANCHOR] " << name << (ok ? "  PASS" : "  FAIL") << "\n"; ok ? ++passes : ++failures; }

struct Out { int rc = 0; std::string text; bool has(const std::string& s) const { return text.find(s) != std::string::npos; } };
static Out tool(std::vector<std::string> a) {
  a.insert(a.begin(), "pchie_anchor"); std::vector<char*> v; for (auto& s : a) v.push_back(s.data());
  std::stringstream o; auto* oc = std::cout.rdbuf(o.rdbuf()); auto* ec = std::cerr.rdbuf(o.rdbuf());
  Out r; r.rc = pchie_anchor_main(int(v.size()), v.data()); std::cout.rdbuf(oc); std::cerr.rdbuf(ec); r.text = o.str(); return r;
}
static std::string field(const std::string& text, const std::string& key) { auto p = text.find(key + "="); if (p == std::string::npos) return ""; auto e = text.find('\n', p); return text.substr(p + key.size() + 1, e - p - key.size() - 1); }
static void show(const std::string& title, const Out& o) { std::cout << "  -- " << title << " (exit " << o.rc << ")\n"; std::stringstream ss(o.text); std::string l; while (std::getline(ss, l)) std::cout << "     " << l << "\n"; }
static pchie::ChainRecord step_record(const std::string& s) { pchie::ChainRecord r; r.kind = pchie::ChainRecordKind::Commit; r.payload.assign(s.begin(), s.end()); return r; }
static pchie::RatchetGenesis gen(const std::string& run, const std::string& weights_hex) { pchie::RatchetGenesis g; g.weights_sha256 = unhex(weights_hex); g.run_id = run; g.harness_version = "bolt-pchie anchoring test"; return g; }
static std::string file_sha(const fs::path& p) { return hex(sha(slurp(p))); }

int main(int argc, char** argv) try {
  const fs::path T = argc > 1 ? fs::path(argv[1]) : fs::current_path() / "anchor-verifier-tmp";
  fs::remove_all(T); fs::create_directories(T);
  const std::string W = hex(sha(std::string("throwaway test weights"))), EMIT = "2026-11-13T06:00:00Z";
  const H32 rd = sha(std::string("resume after step 20, same weights, same context")); const Bytes resume_detail(rd.begin(), rd.end());
  const fs::path log = T / "log", wprod = T / "witness-producer", wremote = T / "witness-remote-test";
  check(tool({"init", log.string(), "test-log"}).rc == 0, "setup: throwaway test log and key created in the build temp folder");
  check(tool({"witness-keygen", wprod.string()}).rc == 0 && tool({"witness-keygen", wremote.string()}).rc == 0, "setup: throwaway producer and remote-test witness keys");
  const std::string prod_pub = trim(slurp(wprod / "public.key"));
  auto emit = [&](const std::string& id, const fs::path& f) { return tool({"emit", log.string(), "--run-id", id, "--label", "Scratch", "--weights", W, "--chain-file", f.string(),
                                                                         "--witness-dir", wprod.string(), "--witness-id", "cortana-producer", "--emitted", EMIT}); };
  auto verify = [&](size_t i, const fs::path& f, const std::string& fp, const fs::path& held, const fs::path& logdir, const std::string& rule = "") {
    std::vector<std::string> a{"verify", logdir.string(), "--anchor-index", std::to_string(i), "--chain-file", f.string(), "--witness-held", held.string(), "--witness-pub", prod_pub, "--witness-id", "cortana-producer"};
    if (!fp.empty()) { a.push_back("--fingerprint"); a.push_back(fp); } if (!rule.empty()) { a.push_back("--rule"); a.push_back(rule); } return tool(a); };

  // ---- runs: three v1 visnav side logs and one ratcheted run with a resume epoch ----
  std::vector<fs::path> v1; for (int r = 0; r < 3; ++r) { fs::path f = T / ("run-v1-" + std::to_string(r) + ".txt"); std::string s; for (int i = 0; i < 30 + 7 * r; ++i) s += "obs run=" + std::to_string(r) + " t=" + std::to_string(i) + " x=" + std::to_string((i * 7 + r) % 13) + " view=ok\n"; spit(f, s); v1.push_back(f); }
  const fs::path rf = T / "run-ratchet.txt";
  { pchie::RatchetHashChain c(gen("run-ratchet", W)); RatchetChainFileWriter w(rf.string(), c);
    for (int i = 0; i < 20; ++i) w.append(step_record("decide t=" + std::to_string(i) + " move=" + std::to_string(i % 4)));
    w.reroot("resume", "run-ratchet", resume_detail);
    for (int i = 20; i < 40; ++i) w.append(step_record("decide t=" + std::to_string(i) + " move=" + std::to_string(i % 4))); w.close(); }
  std::map<std::string, std::string> before; for (auto& f : v1) before[f.string()] = file_sha(f); before[rf.string()] = file_sha(rf);
  std::vector<std::string> ids{"run-v1-0", "run-v1-1", "run-v1-2", "run-ratchet"}; std::vector<fs::path> files{v1[0], v1[1], v1[2], rf}; std::vector<std::string> emitted_fp;
  bool emits_ok = true; for (size_t i = 0; i < 4; ++i) { Out o = emit(ids[i], files[i]); emits_ok &= o.rc == 0; emitted_fp.push_back(field(o.text, "fingerprint")); if (i == 3) show("emit of the ratcheted run", o); }
  check(emits_ok, "setup: four anchors emitted (3 v1, 1 ratcheted), each cosigned by the local producer witness");
  const fs::path held4 = T / "witness-held-size4.txt"; fs::copy_file(wprod / "last_cosigned_sth.txt", held4);
  check(AnchorRecord::parse(Log(log).recs[3]).chain_generation == pchie::kRatchetGenerationId && AnchorRecord::parse(Log(log).recs[0]).chain_generation.empty(),
        "anchor fields: ratcheted anchor carries chain_generation and epoch_id; v1 anchors keep the v1 form");

  // ---- Honest pass, with fingerprint read-back (the number at emit is the number a reviewer re-derives) ----
  { bool ok = true; for (size_t i = 0; i < 4; ++i) { Out o = verify(i, files[i], emitted_fp[i], held4, log); ok &= o.rc == 0 && o.has("OVERALL PASS") && o.has("PASS fingerprint") && o.has("PASS chain_rederive"); if (i == 3) show("honest pass, ratcheted run", o); }
    check(ok, "honest pass: 4 untouched runs verify PASS end to end, fingerprint read back from emit matches"); }

  // ---- Rewrite: one sealed entry altered in a copied chain ----
  { fs::path f = T / "rewrite-v1.txt"; std::string s = slurp(v1[1]); s.replace(s.find("t=5 "), 4, "t=6 "); spit(f, s); Out o = verify(1, f, emitted_fp[1], held4, log);
    fs::path g = T / "rewrite-ratchet.txt"; std::string r = slurp(rf); auto p = r.find("payload=", r.find("step|n=7|")); r[p + 9] = r[p + 9] == '0' ? '1' : '0'; spit(g, r); Out q = verify(3, g, emitted_fp[3], held4, log);
    show("rewrite, ratcheted copy", q);
    check(o.rc == 1 && o.has("FAIL chain_rederive") && o.has("FAIL fingerprint") && q.rc == 1 && q.has("FAIL chain_rederive") && q.has("seal of entry 7 does not re-derive"), "rewrite test: one altered entry FAILS against its anchor (v1 and ratcheted)"); }

  // ---- Recompute: doctored chain regenerated end to end, internally consistent ----
  const fs::path doctored = T / "recompute-ratchet.txt";
  { pchie::RatchetHashChain c(gen("run-ratchet", W)); RatchetChainFileWriter w(doctored.string(), c);
    for (int i = 0; i < 20; ++i) w.append(step_record("decide t=" + std::to_string(i) + " move=" + std::to_string(i == 9 ? 3 : i % 4)));
    w.reroot("resume", "run-ratchet", resume_detail); for (int i = 20; i < 40; ++i) w.append(step_record("decide t=" + std::to_string(i) + " move=" + std::to_string(i % 4))); w.close(); }
  { RatchetFileResult self = rederive_ratchet_chain(doctored.string()); Out o = verify(3, doctored, emitted_fp[3], held4, log); show("recompute, doctored but self-consistent", o);
    check(self.ok && o.rc == 1 && o.has("FAIL chain_rederive") && o.has("FAIL fingerprint"), "recompute test: self-consistent doctored chain still FAILS (head differs from anchored head)"); }
  // Stronger: an attacker holding the log key rewrites the anchor and re-signs; the witness-held head catches it.
  { fs::path la = T / "log-keyholder-attack"; fs::copy(log, la, fs::copy_options::recursive);
    Log lg(la); AnchorRecord a = AnchorRecord::parse(lg.recs[3]); RatchetFileResult d = rederive_ratchet_chain(doctored.string()); a.chain_head_sha256 = hex(d.head); lg.recs[3] = a.canonical();
    std::string all; for (auto& r : lg.recs) all += r + "\n"; spit(la / "anchors.log", all); Log lg2(la); make_sth(lg2, 4);
    Out o = verify(3, doctored, "", held4, la); show("keyholder rewrite, re-signed tree head", o);
    check(o.rc == 1 && o.has("PASS sth_signature") && o.has("PASS chain_rederive") && o.has("FAIL consistency_vs_witness") && o.has("FAIL witness_cosig"), "recompute test (keyholder): re-signed rewrite FAILS against the witness-held head"); }

  // ---- Rewind: truncated log vs witness-held head ----
  { fs::path lr = T / "log-rewound"; fs::copy(log, lr, fs::copy_options::recursive); Log lg(lr); std::string s; for (size_t i = 0; i < 2; ++i) s += lg.recs[i] + "\n"; spit(lr / "anchors.log", s);
    Log lg2(lr); make_sth(lg2, 2); Out o = verify(0, v1[0], emitted_fp[0], held4, lr); show("rewind to 2 entries", o);
    fs::path wc = T / "witness-producer-copy"; fs::copy(wprod, wc, fs::copy_options::recursive); fs::path pk = T / "rewind-packet.txt";
    tool({"witness-packet", lr.string(), "cortana-producer", pk.string()}); Out w = tool({"witness-cosign", wc.string(), "cortana-producer", pk.string(), (T / "rewind-cosig.txt").string()});
    check(o.rc == 1 && o.has("FAIL consistency_vs_witness") && w.rc == 1 && w.has("ROLLBACK"), "rewind test: truncated log FAILS consistency against the witness-held head; witness refuses to cosign"); }

  // ---- Backward fabrication: discarded states and step keys never appear in the chain file or the log ----
  { std::vector<Bytes> keys; std::vector<std::string> L = lines(rf); pchie::RatchetGenesis g = gen("run-ratchet", W); Bytes ck = pchie::RatchetHashChain::genesis_state(g), head = pchie::RatchetHashChain::genesis_head(ck); keys.push_back(ck);
    for (size_t i = 2; i < L.size(); ++i) { std::string tag; auto m = split_fields(L[i], tag);
      if (tag == "step") { pchie::ChainRecord r; r.kind = pchie::ChainRecordKind(std::stoul(m["kind"])); r.payload = unhex(m["payload"]); Bytes mk = pchie::RatchetHashChain::step_key(ck); Bytes s = pchie::RatchetHashChain::seal(mk, head, std::stoull(m["n"]), r); keys.push_back(mk); ck = pchie::RatchetHashChain::advance(ck, s); head = s; }
      else { Bytes c = pchie::RatchetHashChain::epoch_commitment(uint32_t(std::stoul(m["epoch"])), head, std::stoull(m["parent_length"]), m["event"], m["run_id"], unhex(m["detail_sha256"])); ck = pchie::RatchetHashChain::reroot_state(head, c); head = pchie::RatchetHashChain::reroot_head(ck, c); }
      keys.push_back(ck); }
    std::string corpus = slurp(rf) + slurp(log / "anchors.log") + slurp(log / "sth_latest.txt"); for (auto& e : fs::recursive_directory_iterator(log)) if (e.is_regular_file() && e.path().parent_path().filename() != "keys") corpus += slurp(e.path());
    size_t hits = 0; for (auto& k : keys) { hits += corpus.find(hex(k)) != std::string::npos; hits += corpus.find(std::string(k.begin(), k.end())) != std::string::npos; }
    check(hits == 0 && keys.size() > 80, "backward fabrication: none of " + std::to_string(keys.size()) + " chain states or step keys appear in the chain file or the log (one-way steps tested in pchie_ratchet_chain)"); }

  // ---- Epoch confusion: two forks of one parent presented as one straight line ----
  const fs::path pf = T / "fork-parent.txt", fa = T / "fork-a.txt", fb = T / "fork-b.txt"; Bytes ph; size_t pl = 0;
  { pchie::RatchetHashChain c(gen("fork-parent", W)); RatchetChainFileWriter w(pf.string(), c); for (int i = 0; i < 10; ++i) w.append(step_record("parent t=" + std::to_string(i))); w.close(); ph = c.head(); pl = c.length(); }
  auto branch = [&](const fs::path& f, const std::string& id, const std::string& tagname) {
    pchie::RatchetHashChain c(gen("fork-parent", W)); c.restore_head(ph, pl); RatchetChainFileWriter w(f.string(), c); w.copy_entries_from(pf.string());
    H32 d = sha("fork of fork-parent at entry 10 -> " + id); w.reroot("fork", id, Bytes(d.begin(), d.end()));
    for (int i = 0; i < 8; ++i) w.append(step_record(tagname + " t=" + std::to_string(i))); w.close(); };
  branch(fa, "fork-a", "branch a"); branch(fb, "fork-b", "branch b");
  Out ea = emit("fork-a", fa), eb = emit("fork-b", fb); const fs::path held6 = T / "witness-held-size6.txt"; fs::copy_file(wprod / "last_cosigned_sth.txt", held6);
  { auto renumber = [](std::vector<std::string> body, uint64_t start) { for (auto& l : body) { auto p = l.find("|n="), e = l.find('|', p + 1); l = l.substr(0, p) + "|n=" + std::to_string(start++) + l.substr(e); } return body; };
    std::vector<std::string> A = lines(fa), B = lines(fb); std::vector<std::string> Bsteps(B.begin() + 2 + 10 + 1, B.end()), Bepoch_steps(B.begin() + 2 + 10, B.end());
    std::string s1, s2; for (auto& l : A) { s1 += l + "\n"; s2 += l + "\n"; } for (auto& l : renumber(Bsteps, 20)) s1 += l + "\n"; for (auto& l : renumber(Bepoch_steps, 20)) s2 += l + "\n";
    const fs::path f1 = T / "fork-straight-line.txt", f2 = T / "fork-straight-line-with-epoch.txt"; spit(f1, s1); spit(f2, s2);
    Out pa = verify(4, fa, field(ea.text, "fingerprint"), held6, log), pb = verify(5, fb, field(eb.text, "fingerprint"), held6, log);
    Out x1 = verify(5, f1, "", held6, log), x2 = verify(5, f2, "", held6, log), x3 = verify(4, f1, "", held6, log), x4 = verify(5, fa, "", held6, log);
    show("epoch confusion, B steps appended to A", x1); show("epoch confusion, B epoch and steps appended to A", x2);
    check(pa.rc == 0 && pb.rc == 0, "epoch confusion: each fork branch verifies PASS under its own epoch commitment");
    check(x1.rc == 1 && x2.rc == 1 && x3.rc == 1 && x4.rc == 1 && x1.has("FAIL chain_rederive") && x2.has("not one straight line") && x4.has("FAIL epoch"),
          "epoch confusion test: two branches of one parent presented as one line FAIL; branch A against B's anchor FAILS"); }

  // ---- Generation separation ----
  { Out a = verify(0, v1[0], "", held4, log, "v2"), b = verify(3, rf, "", held4, log, "v1"), c = verify(3, v1[0], "", held4, log), d = verify(0, rf, "", held4, log);
    Out ok1 = verify(0, v1[0], "", held4, log), ok2 = verify(3, rf, "", held4, log); show("v1 chain under the ratchet rule", a); show("ratcheted chain under the v1 rule", b);
    check(ok1.rc == 0 && ok2.rc == 0 && a.rc == 1 && b.rc == 1 && c.rc == 1 && d.rc == 1 && a.has("not a ratcheted chain file") && b.has("FAIL chain_generation") && c.has("FAIL chain_generation") && d.has("FAIL chain_generation"),
          "generation separation test: each chain PASSES under its own rule and FAILS cleanly under the other (no crash, exit 1)"); }

  // ---- Thread neutrality, ratcheted generation: 1, 2, 4, 8, 16 workers ----
  { std::vector<std::string> seals_by, heads_by, anchors_by, roots_by; std::vector<size_t> out_of_order;
    for (int workers : {1, 2, 4, 8, 16}) {
      const size_t N = 256; pchie::DeterministicJoin<std::string> join(N); std::atomic<size_t> arrival{0}; std::vector<size_t> arrived(N);
      std::vector<std::thread> pool; for (int w = 0; w < workers; ++w) pool.emplace_back([&, w] { for (size_t i = size_t(w); i < N; i += size_t(workers)) {
        H32 h = sha("step " + std::to_string(i)); for (size_t k = 0; k < 200 + (i * 37) % 300; ++k) h = sha(Bytes(h.begin(), h.end()));
        std::this_thread::yield(); join.deposit(i, "decide t=" + std::to_string(i) + " value=" + hex(h).substr(0, 16)); arrived[i] = arrival++; } });
      for (auto& t : pool) t.join();
      size_t ooo = 0; for (size_t i = 0; i < N; ++i) ooo += arrived[i] != i; out_of_order.push_back(ooo);
      const fs::path f = T / ("thread-neutral-" + std::to_string(workers) + ".txt"), lw = T / ("log-thread-" + std::to_string(workers));
      pchie::RatchetHashChain c(gen("run-thread-neutral", W)); RatchetChainFileWriter wr(f.string(), c); std::string err; int dummy = 0, out = 0;
      join.fold(dummy, [&](int acc, uint64_t id, const std::string& payload) { if (id == 128) wr.reroot("resume", "run-thread-neutral", Bytes(32, 0x22)); wr.append(step_record(payload)); return acc; }, out, err); wr.close();
      std::string seals; for (auto& l : lines(f)) { auto p = l.find("|seal="); if (p != std::string::npos) seals += l.substr(p + 6, 64); else if (l.rfind("epoch|", 0) == 0) seals += l.substr(l.find("|head=") + 6); }
      tool({"init", lw.string(), "test-log"}); Out e = tool({"emit", lw.string(), "--run-id", "run-thread-neutral", "--label", "Scratch", "--weights", W, "--chain-file", f.string(), "--emitted", EMIT});
      seals_by.push_back(hex(sha(seals))); heads_by.push_back(hex(c.head())); anchors_by.push_back(trim(slurp(lw / "anchors.log"))); roots_by.push_back(field(e.text, "root"));
      std::cout << "  -- workers=" << workers << " arrivals_out_of_canonical_order=" << ooo << " seals_sha256=" << seals_by.back().substr(0, 16) << " head=" << heads_by.back().substr(0, 16) << " root=" << roots_by.back().substr(0, 16) << "\n"; }
    bool same = true; for (size_t i = 1; i < seals_by.size(); ++i) same &= seals_by[i] == seals_by[0] && heads_by[i] == heads_by[0] && anchors_by[i] == anchors_by[0] && roots_by[i] == roots_by[0];
    check(same, "thread neutrality, ratcheted generation: identical seals, head, anchor record and tree root at 1/2/4/8/16 workers"); }

  // ---- Witness exchange (file only, test key standing in for a remote witness; nobody is contacted) ----
  { const std::string rpub = trim(slurp(wremote / "public.key")); spit(log / "witness_registry.txt", "remote-test=" + rpub + "\n");
    const fs::path p1 = T / "witness-packet-1.txt", c1 = T / "witness-cosig-1.txt", p2 = T / "witness-packet-2.txt", c2 = T / "witness-cosig-2.txt";
    Out a = tool({"witness-packet", log.string(), "remote-test", p1.string()}); Out b = tool({"witness-cosign", wremote.string(), "remote-test", p1.string(), c1.string()}); Out c = tool({"intake", log.string(), "remote-test", c1.string()});
    std::string s = slurp(v1[2]) + "obs extra\n"; spit(T / "run-v1-extra.txt", s); Out e = emit("run-v1-extra", T / "run-v1-extra.txt");
    Out d = tool({"witness-packet", log.string(), "remote-test", p2.string()}); Out f = tool({"witness-cosign", wremote.string(), "remote-test", p2.string(), c2.string()}); Out g = tool({"intake", log.string(), "remote-test", c2.string()});
    std::string pk = slurp(p1) + slurp(p2); bool leak = false; for (auto& id : ids) leak |= pk.find(id) != std::string::npos; leak |= pk.find(W) != std::string::npos;
    for (auto& r : Log(log).recs) leak |= pk.find(AnchorRecord::parse(r).chain_head_sha256) != std::string::npos;
    std::string bad = trim(slurp(c2)); bad[bad.size() - 3] = bad[bad.size() - 3] == 'a' ? 'b' : 'a'; spit(T / "witness-cosig-bad.txt", bad + "\n"); Out h = tool({"intake", log.string(), "remote-test", (T / "witness-cosig-bad.txt").string()});
    show("second witness packet", d); show("intake", g);
    check(a.rc == 0 && b.rc == 0 && c.has("INTAKE PASS") && e.rc == 0 && d.has("from=6") && f.rc == 0 && g.has("INTAKE PASS") && h.has("INTAKE FAIL") && !leak,
          "witness exchange: packets carry tree heads and a consistency proof only; remote cosign and intake PASS; tampered cosignature FAILS"); }

  // ---- Secret genesis salt: supported, off by default ----
  { Bytes salt(32); for (size_t i = 0; i < salt.size(); ++i) salt[i] = uint8_t(0xA5 ^ (i * 7));  // throwaway test salt
    const fs::path sf = T / "throwaway-genesis-salt.txt", wrong = T / "wrong-genesis-salt.txt", f = T / "run-salted.txt"; spit(sf, hex(salt) + "\n"); Bytes ws = salt; ws[0] ^= 1; spit(wrong, hex(ws) + "\n");
    { pchie::RatchetHashChain c(pchie::RatchetGenesis::with_secret_salt(gen("run-salted", W), salt)); RatchetChainFileWriter w(f.string(), c);
      for (int i = 0; i < 12; ++i) w.append(step_record("salted t=" + std::to_string(i))); w.reroot("resume", "run-salted", resume_detail); for (int i = 12; i < 17; ++i) w.append(step_record("salted t=" + std::to_string(i))); w.close(); }
    Out nosalt_emit = emit("run-salted", f);
    Out e = tool({"emit", log.string(), "--run-id", "run-salted", "--label", "Scratch", "--weights", W, "--chain-file", f.string(), "--genesis-salt-file", sf.string(), "--witness-dir", wprod.string(), "--witness-id", "cortana-producer", "--emitted", EMIT});
    const std::string idx = field(e.text, "anchor_index"), fp = field(e.text, "fingerprint");
    Out ok = tool({"verify", log.string(), "--anchor-index", idx, "--chain-file", f.string(), "--genesis-salt-file", sf.string(), "--fingerprint", fp});
    Out no = tool({"verify", log.string(), "--anchor-index", idx, "--chain-file", f.string(), "--fingerprint", fp});
    Out bad = tool({"verify", log.string(), "--anchor-index", idx, "--chain-file", f.string(), "--genesis-salt-file", wrong.string(), "--fingerprint", fp});
    show("salted chain, verified with its salt", ok); show("salted chain, verified without the salt", no);
    const std::string corpus = slurp(f) + slurp(log / "anchors.log");
    check(nosalt_emit.rc != 0 && nosalt_emit.has("salt is required") && e.rc == 0 && field(e.text, "chain_generation") == pchie::kRatchetSaltedGenerationId && ok.rc == 0 && ok.has("OVERALL PASS") &&
          no.rc == 1 && no.has("salt is required") && bad.rc == 1 && bad.has("does not match the chain's salt commitment") && corpus.find(hex(salt)) == std::string::npos,
          "secret-salt mode: anchors and verifies only with its salt; without it or with a wrong salt FAILS cleanly; salt never written");
    check(ratchet_file_generation(rf.string()) == pchie::kRatchetGenerationId && slurp(rf).find("salt_commitment") == std::string::npos && rederive_ratchet_chain(rf.string()).ok,
          "secret-salt mode is off by default: the default ratcheted chain stays recomputable with no salt"); }

  // ---- Fingerprint covers chain_generation for ratcheted records; the v1 rule is unchanged ----
  { const std::string h(64, 'b'); const std::string f1 = fingerprint_for(h, "golden-run", ""), f2 = fingerprint_for(h, "golden-run", pchie::kRatchetGenerationId), f3 = fingerprint_for(h, "golden-run", pchie::kRatchetSaltedGenerationId);
    RatchetFileResult rr = rederive_ratchet_chain(rf.string()); Out v1rule = verify(3, rf, fingerprint(hex(rr.head), "run-ratchet"), held4, log); Out v2rule = verify(3, rf, fingerprint_for(hex(rr.head), "run-ratchet", pchie::kRatchetGenerationId), held4, log);
    std::cout << "  -- same head and run id: v1=" << f1 << "\n     v2-ratchet=" << f2 << "\n     v2-ratchet-salted=" << f3 << "\n";
    check(f1 == "10946 64075 96542 49169 15837 49247 60686 95747 96227 50449 55335 59234" && f2 != f1 && f3 != f1 && f3 != f2 && f2.size() == 71 && v2rule.rc == 0 && v2rule.has("covers chain_generation") && v1rule.rc == 1 && v1rule.has("FAIL fingerprint"),
          "fingerprint differs across generations; v1 fingerprint pinned unchanged; a v1-rule number does not pass for a ratcheted anchor"); }

  // ---- Witness policy: defined three-way set, bench 2-of-2 (jd-local + cortana-producer), open for N ----
  { const fs::path pdir = argc > 2 ? fs::path(argv[2]) : fs::path("anchoring/policy"); const fs::path bench = pdir / "bench.txt", three = pdir / "three-way.txt";
    const fs::path wjd = T / "witness-jd-local", wsrv = T / "witness-server-a";
    tool({"witness-keygen", wjd.string()}); tool({"witness-keygen", wsrv.string()});
    spit(log / "witness_registry.txt", "jd-local=" + trim(slurp(wjd / "public.key")) + "\ncortana-producer=" + prod_pub + "\nelfie=UNREGISTERED\nserver-a=" + trim(slurp(wsrv / "public.key")) + "\n");
    auto vp = [&](const std::string& idx, const fs::path& pol) { return tool({"verify", log.string(), "--anchor-index", idx, "--policy", pol.string()}); };
    spit(T / "run-policy-1.txt", "obs policy t=0\nobs policy t=1\n"); Out e1 = emit("run-policy-1", T / "run-policy-1.txt"); const std::string i1 = field(e1.text, "anchor_index");
    Out missing = vp(i1, bench); show("bench policy, only Cortana has cosigned", missing);
    Out c1 = tool({"cosign-local", log.string(), wjd.string(), "jd-local"});
    Out pass = vp(i1, bench), threeway = vp(i1, three); show("bench policy after the jd-local cosign", pass); show("three-way policy (Elfie not enrolled)", threeway);
    const fs::path jc = log / "cosigs" / (field(e1.text, "tree_size") + "_jd-local.txt"); const std::string keep = slurp(jc); std::string t = trim(keep); t[t.size() - 5] = t[t.size() - 5] == 'a' ? 'b' : 'a'; spit(jc, t + "\n");
    Out tampered = vp(i1, bench); spit(jc, keep);
    spit(T / "run-policy-2.txt", "obs policy t=2\n"); Out e2 = emit("run-policy-2", T / "run-policy-2.txt"); Out c2 = tool({"cosign-local", log.string(), wjd.string(), "jd-local"}); Out pass2 = vp(field(e2.text, "anchor_index"), bench);
    spit(T / "policy-four-servers.txt", "policy=four-servers\nwitnesses=jd-local,cortana-producer,elfie,server-a\nthreshold=3\n"); Out c3 = tool({"cosign-local", log.string(), wsrv.string(), "server-a"}); Out four = vp(field(e2.text, "anchor_index"), T / "policy-four-servers.txt");
    check(missing.rc == 1 && missing.has("FAIL witness_policy") && missing.has("MISSING jd-local"), "witness policy: bench FAILS when the jd-local cosigner is missing");
    check(c1.rc == 0 && pass.rc == 0 && pass.has("PASS witness_policy") && pass.has("valid=2/3") && threeway.rc == 1 && threeway.has("not signed: elfie"),
          "witness policy: bench 2-of-2 (jd-local + cortana-producer) PASSES; the defined three-way policy still needs Elfie");
    check(tampered.rc == 1 && tampered.has("FAIL witness_policy") && c2.rc == 0 && c2.has("tree_size=") && pass2.rc == 0, "witness policy: a tampered jd-local cosignature FAILS; the next cosign checks consistency from its last head and PASSES");
    check(c3.rc == 0 && four.rc == 0 && four.has("valid=3/4"), "witness policy: open for N witnesses (four listed, threshold 3) PASSES"); }

  // ---- Key substitution: a witness pins the log key and refuses a packet signed under any other key ----
  { const fs::path log2 = T / "log-substitute"; check(tool({"init", log2.string(), "test-log"}).rc == 0, "setup: a second throwaway log with the same log id and its own key");
    Out e2 = tool({"emit", log2.string(), "--run-id", "substitute", "--label", "Scratch", "--weights", W, "--chain-file", v1[0].string(), "--emitted", EMIT});
    const fs::path ps = T / "packet-substitute.txt", cs = T / "cosig-substitute.txt"; Out pk = tool({"witness-packet", log2.string(), "remote-test", ps.string()});
    Out sub = tool({"witness-cosign", wremote.string(), "remote-test", ps.string(), cs.string()}); show("cosign a packet from a log with a different key", sub);
    const bool pinned = fs::exists(wremote / "trusted_log_pub.txt") && trim(slurp(wremote / "trusted_log_pub.txt")) == trim(slurp(log / "keys" / "public.key"));
    check(e2.rc == 0 && pk.rc == 0 && sub.rc == 1 && sub.has("LOG KEY CHANGED") && !fs::exists(cs) && pinned,
          "key substitution: the witness pinned the log key on first contact and refuses a packet under a different key");
    // Pinned ahead of time with witness-trust: refused on first contact too; the right log still cosigns.
    const fs::path wf = T / "witness-pinned-ahead"; tool({"witness-keygen", wf.string()});
    Out tr = tool({"witness-trust", wf.string(), trim(slurp(log / "keys" / "public.key"))});
    const fs::path pg = T / "packet-pinned-good.txt", pb = T / "packet-pinned-bad.txt";
    tool({"witness-packet", log2.string(), "pinned-ahead", pb.string()}); tool({"witness-packet", log.string(), "pinned-ahead", pg.string()});
    Out bad = tool({"witness-cosign", wf.string(), "pinned-ahead", pb.string(), (T / "cosig-pinned-bad.txt").string()});
    Out good = tool({"witness-cosign", wf.string(), "pinned-ahead", pg.string(), (T / "cosig-pinned-good.txt").string()});
    Out retr = tool({"witness-trust", wf.string(), trim(slurp(log2 / "keys" / "public.key"))});
    check(tr.rc == 0 && bad.rc == 1 && bad.has("LOG KEY CHANGED") && good.rc == 0 && retr.rc != 0,
          "key substitution: a key pinned with witness-trust refuses another log on first contact, accepts its own, and cannot be silently re-pinned");
    // Witness state from before pinning: a different key is adopted only if it signed the head already held.
    const fs::path wl = T / "witness-legacy"; fs::copy(wprod, wl, fs::copy_options::recursive); fs::remove(wl / "trusted_log_pub.txt");
    const fs::path pl2 = T / "packet-legacy.txt"; tool({"witness-packet", log2.string(), "cortana-producer", pl2.string()});
    Out leg = tool({"witness-cosign", wl.string(), "cortana-producer", pl2.string(), (T / "cosig-legacy.txt").string()});
    check(leg.rc == 1 && leg.has("LOG KEY CHANGED"), "key substitution: a witness with held state but no pin refuses a key that did not sign its held head"); }

  // ---- Witness ids are names, not paths ----
  { Out a = tool({"witness-cosign", wremote.string(), "../escape", (T / "witness-packet-2.txt").string(), (T / "cosig-escape.txt").string()});
    Out b = tool({"emit", log.string(), "--run-id", "bad-id", "--label", "Scratch", "--weights", W, "--chain-file", v1[0].string(), "--witness-dir", wprod.string(), "--witness-id", "a/b", "--emitted", EMIT});
    Out c = tool({"witness-packet", log.string(), std::string(65, 'w'), (T / "packet-long-id.txt").string()});
    Out d = tool({"cosign-local", log.string(), wprod.string(), ".hidden"});
    check(a.rc != 0 && b.rc != 0 && c.rc != 0 && d.rc != 0 && !fs::exists(T / "cosig-escape.txt") && !fs::exists(T / "escape"),
          "witness ids: path separators, '..', leading dots and over-long ids are refused before any file is touched");
    check(witness_id_ok("cortana-producer") && witness_id_ok("jd-local") && witness_id_ok("server_a.2") && !witness_id_ok("") && !witness_id_ok("a\\b") && !witness_id_ok("-x") && !witness_id_ok("a..b"),
          "witness ids: the allowlist accepts the ids in use and rejects the rest"); }

  // ---- Concurrent signing never reuses a one-time leaf ----
  { const fs::path kd = T / "concurrent-key"; fs::create_directories(kd); keygen(kd, "pchie.anchor.test.concurrent");
    const fs::path kp = kd / "signing.key"; const Bytes pub = unhex(trim(slurp(kd / "public.key")));
    const int threads = 8, per = 4; std::vector<std::vector<Bytes>> sigs(threads); std::vector<std::vector<std::string>> msgs(threads); std::atomic<int> errors{0};
    std::vector<std::thread> pool;
    for (int t = 0; t < threads; ++t) pool.emplace_back([&, t] { for (int i = 0; i < per; ++i) { try { std::string m = "concurrent " + std::to_string(t) + "/" + std::to_string(i);
          sigs[t].push_back(sign_persist(kp, m)); msgs[t].push_back(m); } catch (...) { ++errors; } } });
    for (auto& th : pool) th.join();
    std::set<uint32_t> qs; bool all_verify = true; size_t n = 0;
    for (int t = 0; t < threads; ++t) for (size_t i = 0; i < sigs[t].size(); ++i) { const Bytes& sg = sigs[t][i]; ++n;
      qs.insert((uint32_t(sg[0]) << 24) | (uint32_t(sg[1]) << 16) | (uint32_t(sg[2]) << 8) | uint32_t(sg[3]));
      all_verify &= hbs_verify(pub, Bytes(msgs[t][i].begin(), msgs[t][i].end()), sg); }
    HbsKey after = load_key(kp); const uint32_t next_q = after.next_q; after.wipe();
    bool no_temp = true; for (auto& e : fs::directory_iterator(kd)) { const std::string nm = e.path().filename().string(); no_temp &= nm.find(".tmp") == std::string::npos; }
    std::cout << "  -- concurrent signing: " << n << " signatures, " << qs.size() << " distinct leaves, next_q=" << next_q << ", errors=" << errors.load() << "\n";
    check(errors == 0 && n == size_t(threads * per) && qs.size() == n && next_q == uint32_t(n) && all_verify && no_temp,
          "concurrent signing: 8 threads x 4 signatures under the key lock use 32 distinct leaves, all verify, next_q = 32, no temp files left"); }

  // ---- An exhausted key refuses before it reserves anything ----
  { const fs::path kd = T / "exhausted-key"; fs::create_directories(kd); keygen(kd, "pchie.anchor.test.exhausted");
    const fs::path kp = kd / "signing.key";
    { HbsKey k = load_key(kp); k.next_q = Hbs::kLeaves; save_key(kp, k); k.wipe(); }
    const std::string before_sha = file_sha(kp);
    bool threw = false; try { sign_persist(kp, "after exhaustion"); } catch (const std::exception&) { threw = true; }
    HbsKey after = load_key(kp); const uint32_t nq = after.next_q; after.wipe();
    check(threw && nq == Hbs::kLeaves && file_sha(kp) == before_sha,
          "exhausted key: signing refuses with every leaf used, and the key file is left exactly as it was"); }

  // ---- Neutrality: anchoring never changes run artifacts, and the v1 record form is byte-for-byte the old one ----
  { bool same = true; for (auto& kv : before) same &= file_sha(kv.first) == kv.second;
    AnchorRecord a; a.run_id = "golden-run"; a.label = "Scratch"; a.weights_sha256 = std::string(64, 'a'); a.chain_head_sha256 = std::string(64, 'b'); a.chain_length = 3; a.prev_anchor_sha256 = std::string(64, '0'); a.emitted_utc = "2026-11-13T00:00:00Z";
    check(same, "neutrality: run chain files byte-identical after emit and verify");
    check(hex(a.record_hash()) == "3a1cb1fa7e8cf4851771926397e53b40b53e708d66cacd6f85bdd37e49a99606" && fingerprint(std::string(64, 'b'), "golden-run") == "10946 64075 96542 49169 15837 49247 60686 95747 96227 50449 55335 59234",
          "neutrality: v1 anchor record hash and fingerprint equal the values from the untouched tree"); }

  std::cout << "ANCHOR-VERIFIER: " << passes << " passed, " << failures << " failed\n";
  if (!failures) fs::remove_all(T);  // throwaway keys and logs go away on a clean pass
  return failures ? 1 : 0;
} catch (const std::exception& e) { std::cout << "ERROR " << e.what() << "\n"; return 2; }
