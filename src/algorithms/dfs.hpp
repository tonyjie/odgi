#pragma once

#include <handlegraph/handle_graph.hpp>
#include <vector>
#include <set>
#include <deque>
#include "hash_map.hpp"

namespace odgi {
namespace algorithms {
    
    
using namespace std;
using namespace handlegraph;


void dfs(
    const HandleGraph& graph,
    const function<void(const handle_t&)>& handle_begin_fn,  // called when node orientation is first encountered
    const function<void(const handle_t&)>& handle_end_fn,    // called when node orientation goes out of scope
    const function<bool(void)>& break_fn,                    // called to check if we should stop the DFS; we stop when true is returned.
    const function<void(const edge_t&)>& edge_fn,            // called when an edge is encountered
    const function<void(const edge_t&)>& tree_fn,            // called when an edge forms part of the DFS spanning tree
    const function<void(const edge_t&)>& edge_curr_fn,       // called when we meet an edge in the current tree component
    const function<void(const edge_t&)>& edge_cross_fn,      // called when we meet an edge in an already-traversed tree component
    const vector<handle_t>& sources,                         // start only at these node traversals
    const ska::flat_hash_set<handle_t>& sinks);                   // when hitting a sink, don't keep walking

void dfs(const HandleGraph& graph,
         const function<void(const handle_t&)>& handle_begin_fn,
         const function<void(const handle_t&)>& handle_end_fn,
         const vector<handle_t>& sources,
         const ska::flat_hash_set<handle_t>& sinks);

void dfs(const HandleGraph& graph,
         const function<void(const handle_t&)>& handle_begin_fn,
         const function<void(const handle_t&)>& handle_end_fn,
         const function<bool(void)>& break_fn);

}
}


