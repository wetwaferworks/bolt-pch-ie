// Invalid KV reads are refused in Release builds too; a throwing token source must not leak cloned pages.
#include "pchie/pchie_host.h"
#include "pchie/pchie_kv_pools.h"
#include <cstdio>
#include <stdexcept>
#include <string>
using namespace pchie;
static int pass=0, fail=0;
static void check(bool ok,const char* n){ std::printf("[BOUNDS] %s %s\n",n,ok?"PASS":"FAIL"); ok?++pass:++fail; }
template<class F> static bool refused(F f){ try{f();}catch(const std::out_of_range&){return true;}return false; }
int main(){
    KVPagePool p(1,1,8,8); KVPagePool::Table t;
    p.k_write(t,0,0)[0]=3; p.v_write(t,0,0)[0]=4;
    check(p.k_read(t,0,0)[0]==3 && p.v_read(t,0,0)[0]==4,"valid KV read unchanged");
    check(refused([&]{p.k_read(t,1,0);}),"bad layer refused");
    check(refused([&]{p.v_read(t,0,16);}),"position past table refused");
    p.trim_before(t,16);
    check(refused([&]{p.k_read(t,0,0);}),"trimmed key read refused (r07)");
    check(refused([&]{p.v_read(t,0,0);}),"trimmed value read refused");
    p.release(t); p.release(t);
    check(p.live_pages()==0,"release same table twice is safe");
    p.k_write(t,0,0); auto stale=t; p.release(t);
    check(refused([&]{p.release(stale);}),"release stale plain copy refused before refcount underflow");
    check(refused([&]{p.k_write(t,0,UINT32_MAX);}),"huge position refused before resizing table");
    bool zero=false; try{TimingHandshake c(0,0);}catch(const std::invalid_argument&){zero=true;}
    check(zero,"zero ticks per token refused");
    HostSession h(0,1,1,[](const TokenRecord&,uint32_t)->GenOut{throw std::runtime_error("test source failure");});
    const auto before=h.kv.free_blocks(); bool threw=false;
    try{h.generate();}catch(const std::runtime_error&){threw=true;}
    check(threw && h.kv.free_blocks()==before,"throwing token source releases cloned pages");
    check(h.tree.nodes.size()==1 && h.clock.now()==0,"throw leaves tree and clock at parent");
    std::printf("[BOUNDS] %d PASS / %d FAIL\n",pass,fail); return fail?1:0;
}
