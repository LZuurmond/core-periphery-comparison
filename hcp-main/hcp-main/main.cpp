//
// Written by Austin Polanco on 15 JUN 22.
//

#include <iostream>
#include "parameters.h"
#include "hierarchical_model.h"
#include <chrono>
#include <fstream>
#include <ctime>
#include <queue>
#include <vector>
#include <algorithm>


// Structure to hold a complete snapshot of a solution
struct SolutionState {
    std::vector<uint64_t> groups;
    std::vector<long long> hcg_edges;
    std::vector<long long> hcg_pairs;
    std::vector<long long> group_size;
    double energy;
    std::size_t num_groups;

    /* Comparator for min-heap
    *  we want the smallest energy at the top so we can easily replace it
    *  if we find a better solution
    */
    bool operator>(const SolutionState& other) const {
        return energy > other.energy;
    }
};

int main(int argc, char* argv[]) {

    std::string config_file = argv[1];
    parameters params(config_file);

    if (params.get_error_status() == 1){
        return EXIT_FAILURE;
    }


    hierarchical_model hcp(params);

    hcp.print_hcg_pairs();
    std::cout<<std::endl;
    hcp.print_hcg_edges();
    std::cout<<std::endl;

    std::vector<std::vector<uint64_t>> intermediate_states;
    std::vector<std::vector<long long>> hcg_edges;
    std::vector<std::vector<long long>> hcg_pairs;
    std::vector<std::vector<long long>> group_size;
    std::vector<double> energies;
    std::vector<std::size_t> num_groups;

    // Priority queue for top-k solutions
    std::priority_queue<SolutionState, std::vector<SolutionState>, std::greater<SolutionState>> top_k_heap;
    
    // get target number of solutions
    std::size_t k_target = params.get_num_solutions();

    // get max number of iterations
    long num_itrs = params.get_max_itr();

    // hard-coded burn-in and thinning from the original paper
    // this is to avoid k identical copies of the same peak
    long burn_in = 10000000;
    long thinning = 1500;


    for(long i = 0; i < num_itrs; ++i){
        // perform MCMC step
        hcp.get_groups();

        // only sample after burn-in and at specific intervals to ensure diversity
        if ((i > burn_in) && (i % thinning == 0)) {
            
            // check if we should add this state to our Top K
            bool should_add = false;
            if (top_k_heap.size() < k_target) {
                should_add = true;
            }
            else if (hcp.loglike > top_k_heap.top().energy) {
                should_add = true;
                top_k_heap.pop(); // remove the worst of the current top K
            }

            if (should_add) {
                SolutionState current_state;
                current_state.groups = hcp.g;
                current_state.hcg_edges = hcp.hcg_edges;
                current_state.hcg_pairs = hcp.hcg_pairs;
                current_state.group_size = hcp.group_size;
                current_state.energy = hcp.loglike;
                current_state.num_groups = hcp.num_groups;

                top_k_heap.push(current_state);
            }
        }

        if(i%10000000==0){
            std::cout<<"-----------------------------------------------------"<<std::endl;
            auto curr = std::chrono::system_clock::now();
            auto tm = std::chrono::system_clock::to_time_t(curr);
            std::cout<<"time: "<< std::put_time(std::localtime(&tm), "%c %Z")<<std::endl;
            std::cout<<"iteration: "<<i<<" energy: "<<hcp.loglike<<std::endl;

            // print status of heap
            if (!top_k_heap.empty()) {
                std::cout << "Current Top-K Worst Energy: " << top_k_heap.top().energy << std::endl;
            }

            hcp.print_hcg_pairs();
            std::cout<<std::endl;
            hcp.print_hcg_edges();
            std::cout<<std::endl;
            hcp.print_group_size();
            std::cout<<std::endl;
        }
    }

    std::cout << "Writing Top " << top_k_heap.size() << " solutions to file." << std::endl;
    
    //  transfer heap to vector for sorting
    std::vector<SolutionState> sorted_solutions;
    while (!top_k_heap.empty()) {
        sorted_solutions.push_back(top_k_heap.top());
        top_k_heap.pop();
    }

    // Sort descending (best energy first)
    std::sort(sorted_solutions.begin(), sorted_solutions.end(),
        [](const SolutionState& a, const SolutionState& b) {
            return a.energy > b.energy;
    });

    std::string filename = params.get_saved_data_name();
    std::string filepath = params.get_save_dir();
    std::ofstream output_groups(filepath+filename+"_configs.txt");
    std::ofstream output_ngroups(filepath+filename+"_num_groups.txt");
    std::ofstream output_group_size(filepath+filename+"_group_size.txt");
    std::ofstream output_edges(filepath+filename+"_edges.txt");
    std::ofstream output_pairs(filepath+filename+"_pairs.txt");
    std::ofstream output_ll(filepath+filename+"_ll.txt");

    for (const auto& sol : sorted_solutions) {
        // output groups
        for (size_t l = 0; l < sol.groups.size(); ++l) {
            output_groups << sol.groups[l] << " ";
        }
        output_groups << std::endl;

        // output edges
        for (size_t mu = 0; mu < sol.hcg_edges.size(); ++mu) {
            output_edges << sol.hcg_edges[mu] << " ";
            output_pairs << sol.hcg_pairs[mu] << " ";
            output_group_size << sol.group_size[mu] << " ";
        }
        output_edges << std::endl;
        output_pairs << std::endl;
        output_group_size << std::endl;

        // Output energy and num groups
        output_ll << sol.energy << std::endl;
        output_ngroups << sol.num_groups << std::endl;
    }

    output_groups.close();
    output_edges.close();
    output_pairs.close();
    output_group_size.close();
    output_ll.close();

    std::cout<<"Simulation data saved successfully."<<std::endl;

    return 0;
}
