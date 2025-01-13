// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause
//This file is heavily inspired (if not entirely copied from) the ginkgo's project examples:
/* https://github.com/ginkgo-project/ginkgo/tree/develop/examples */

#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>
#include <ginkgo/ginkgo.hpp>


int main(int argc, char* argv[])
{
    // Some shortcuts
    using _TYPE_ = double;
    using IndexType = int;
    using vec = gko::matrix::Dense<_TYPE_>;
    using mtx = gko::matrix::Csr<_TYPE_, IndexType>;
    using cg = gko::solver::Cg<_TYPE_>;
    using mg = gko::solver::Multigrid;
    using pgm = gko::multigrid::Pgm<_TYPE_, IndexType>;

    // Print version information
    std::cout <<"\n"<< gko::version_info::get() << std::endl;

    const auto executor_string = argc >= 2 ? argv[1] : "reference";
    const double rtol = argc >= 3 ? std::stod(argv[2]) : 5e-4;

    // Figure out where to run the code
    std::map<std::string, std::function<std::shared_ptr<gko::Executor>()>>
        exec_map{
            {"omp", [] { return gko::OmpExecutor::create(); }},
            {"cuda",
             [] {
                 return gko::CudaExecutor::create(0,
                                                  gko::OmpExecutor::create());
             }},
            {"hip",
             [] {
                 return gko::HipExecutor::create(0, gko::OmpExecutor::create());
             }},
            {"dpcpp",
             [] {
                 return gko::DpcppExecutor::create(
                     0, gko::ReferenceExecutor::create());
             }},
            {"reference", [] { return gko::ReferenceExecutor::create(); }}};

    // executor where Ginkgo will perform the computation (device ?)
    const auto exec = exec_map.at(executor_string)();  // throws if not valid

    // Query the device properties
    std::cout<<"\nBACKEND SELECTED: "<< executor_string <<"\n";
    std::cout<<"Executor: "<< exec->get_description() <<"\n";
    std::cout<<"rtol= "<< rtol <<"\n";
 
    // Read matrix and rhs
    //auto A = share(gko::read<mtx>(std::ifstream("data/aij_51840.mtx"), exec));
    //auto b = share(gko::read<vec>(std::ifstream("data/rhs_51840.mtx"), exec));

    // Read matrix and rhs
    std::cout<<"\nReading matrix ...\n";
    auto A = share(gko::read<mtx>(std::ifstream("../data/aij_2592000.mtx"), exec));
    std::cout<<"OK! Reading rhs ...\n";
    auto b = share(gko::read<vec>(std::ifstream("../data/rhs_2592000.mtx"), exec));

    std::cout<<"\nCreate null initial guess x and loading onto device ...\n";
    // Create initial guess as 0
    gko::size_type size = A->get_size()[0];
    auto host_x = vec::create(exec->get_master(), gko::dim<2>(size, 1));
    for (auto i = 0; i < size; i++) {
        host_x->at(i, 0) = 0.;
    }
    auto x = vec::create(exec);
    //Send to device
    x->copy_from(host_x);

    std::cout<<"Evaluating initial residual...\n";
    // Calculate initial residual
    auto r = share(gko::read<vec>(std::ifstream("../data/rhs_2592000.mtx"), exec));
    auto one = gko::initialize<vec>({1.0}, exec); 
    auto neg_one = gko::initialize<vec>({-1.0}, exec);
    auto initres = gko::initialize<vec>({0.0}, exec);
    A->apply(one, x, neg_one, r); //r=Ax-r (with r=b)
    r->compute_norm2(initres);

    std::cout<<"Creating multigrid factory...\n";
    // Create multigrid factory
    std::shared_ptr<gko::LinOpFactory> multigrid_gen;
    multigrid_gen =
        mg::build()
            .with_mg_level(pgm::build().with_deterministic(true))
            .with_criteria(gko::stop::Iteration::build().with_max_iters(1u))
            .on(exec);
    const gko::remove_complex<_TYPE_> tolerance = rtol;
    auto solver_gen =
        cg::build()
            .with_criteria(gko::stop::Iteration::build().with_max_iters(100u),
                           gko::stop::ResidualNorm<_TYPE_>::build()
                               .with_baseline(gko::stop::mode::rhs_norm)
                               .with_reduction_factor(tolerance))
            .with_preconditioner(multigrid_gen)
            .on(exec);

    std::cout<<"Creating solver ...\n";
    // Create solver
    std::chrono::nanoseconds gen_time(0);
    auto gen_tic = std::chrono::steady_clock::now();
    auto solver = solver_gen->generate(A);
    exec->synchronize();
    auto gen_toc = std::chrono::steady_clock::now();
    gen_time +=
        std::chrono::duration_cast<std::chrono::nanoseconds>(gen_toc - gen_tic);

    // Add logger
    std::shared_ptr<const gko::log::Convergence<_TYPE_>> logger =
        gko::log::Convergence<_TYPE_>::create();
    solver->add_logger(logger);

    std::cout<<"\nAll ready... Solving !\n";

    // Solve system
    exec->synchronize();
    std::chrono::nanoseconds time(0);
    auto tic = std::chrono::steady_clock::now();
    solver->apply(b, x);
    exec->synchronize();
    auto toc = std::chrono::steady_clock::now();
    time += std::chrono::duration_cast<std::chrono::nanoseconds>(toc - tic);
    std::cout<<"OK! results:\n\n";

    // Calculate residual
    auto res = gko::as<vec>(logger->get_residual_norm());

    std::cout << "Initial residual norm sqrt(r^T r): \n";
    write(std::cout, initres);
    std::cout << "Final residual norm sqrt(r^T r): \n";
    write(std::cout, res);

    // Print solver statistics
    std::cout << "CG iteration count:     " << logger->get_num_iterations()
              << std::endl;
    std::cout << "CG generation time [ms]: "
              << static_cast<double>(gen_time.count()) / 1e9 << std::endl;
    std::cout << "CG execution time [ms]: "
              << static_cast<double>(time.count()) / 1e9 << std::endl;
    std::cout << "CG execution time per iteration[ms]: "
              << static_cast<double>(time.count()) / 1e6 /
                     logger->get_num_iterations()
              << std::endl;
} 
