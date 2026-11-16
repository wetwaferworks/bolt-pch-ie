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
    {   // freed pages come back lowest id first, whatever order the tables were released in
        KVPagePool q(1,1,8,8); KVPagePool::Table a,b,c;
        for(uint32_t pos=0;pos<4*KV_PAGE_TOKENS;pos+=KV_PAGE_TOKENS) q.k_write(a,0,pos);
        for(uint32_t pos=0;pos<2*KV_PAGE_TOKENS;pos+=KV_PAGE_TOKENS) q.k_write(b,0,pos);
        q.truncate(a,KV_PAGE_TOKENS); q.release(b); q.trim_before(a,KV_PAGE_TOKENS); q.release(a);
        bool order=q.live_pages()==0;
        for(uint32_t i=0;i<6;++i){ q.k_write(c,0,i*KV_PAGE_TOKENS); order=order && q.page_of(c,i*KV_PAGE_TOKENS)==int32_t(i); }
        check(order,"free list order after truncate, release and trim is lowest id first");
    }
    std::printf("[BOUNDS] %d PASS / %d FAIL\n",pass,fail); return fail?1:0;
}
