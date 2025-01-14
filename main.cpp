// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause
// This file is heavily inspired (if not entirely copied from) the ginkgo's project examples:
/* https://github.com/ginkgo-project/ginkgo/tree/develop/examples */

#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>
#include <ginkgo/ginkgo.hpp>
#include <mynvtx.h>

// Some shortcuts
using _TYPE_ = double;
using _STYPE_ = int;
using vec = gko::matrix::Dense<_TYPE_>;
using mtx = gko::matrix::Csr<_TYPE_, _STYPE_>;
using conjugate_gradient = gko::solver::Cg<_TYPE_>;
using multigrid = gko::solver::Multigrid;
using parallel_graph_matching = gko::multigrid::Pgm<_TYPE_, _STYPE_>;
using jacobi_preconditioner = gko::preconditioner::Jacobi<_TYPE_, _STYPE_>;
using incomplete_cholesky_preconditioner = gko::preconditioner::Ic<gko::solver::LowerTrs<_TYPE_>>;
using incomplete_cholesky_factorisation = gko::factorization::Ic<_TYPE_, _STYPE_>;

int main(int argc, char *argv[])
{
    // Resolution parameters
    auto max_block_size_jacobi = 4u;
    bool use_storage_optim_jacobi = true;
    auto n_smooth = 2u;
    _TYPE_ relax_smooth = 0.9;
    bool pgm_deterministic = true;
    auto max_levels = 5u;
    auto n_v_cycles = 1u;

    // ---- Read user input and select device ----

    // Print version information
    std::cout << "\n" << gko::version_info::get() << std::endl;

    const auto executor_string = argc >= 2 ? argv[1] : "reference";

    // Figure out where to run the code
    std::map<std::string, std::function<std::shared_ptr<gko::Executor>()>>
        exec_map{
            {"omp", []
             { return gko::OmpExecutor::create(); }},
            {"cuda",
             []
             {
                 return gko::CudaExecutor::create(0,
                                                  gko::OmpExecutor::create());
             }},
            {"hip",
             []
             {
                 return gko::HipExecutor::create(0, gko::OmpExecutor::create());
             }},
            {"dpcpp",
             []
             {
                 return gko::DpcppExecutor::create(
                     0, gko::ReferenceExecutor::create());
             }},
            {"reference", []
             { return gko::ReferenceExecutor::create(); }}};

    // executor where Ginkgo will perform the computation (device ?)
    const auto exec = exec_map.at(executor_string)(); // throws if not valid
    // Query the device properties
    std::cout << "\nBACKEND SELECTED: " << executor_string << "\n";
    std::cout << "Executor: " << exec->get_description() << "\n";

    // ---- Read matrix and evaluate initial error----

    // Read matrix and rhs
    // auto A = share(gko::read<mtx>(std::ifstream("data/aij_51840.mtx"), exec));
    // auto b = share(gko::read<vec>(std::ifstream("data/rhs_51840.mtx"), exec));

    // Read matrix and rhs
    std::cout << "\nReading matrix ...\n";
    auto A = share(gko::read<mtx>(std::ifstream("../data/aij_2592000.mtx"), exec));
    std::cout << "Reading rhs ...\n";
    auto b = share(gko::read<vec>(std::ifstream("../data/rhs_2592000.mtx"), exec));

    std::cout << "\nCreate null initial guess x and loading onto device ...\n";
    
    // Create initial guess as 0
    gko::size_type size = A->get_size()[0];
    auto host_x = vec::create(exec->get_master(), gko::dim<2>(size, 1));
    for (auto i = 0; i < size; i++)
    {
        host_x->at(i, 0) = 0.;
    }
    auto x = vec::create(exec);
    // Send to device
    x->copy_from(host_x);

    std::cout << "Evaluating initial residual...\n";
    // Calculate initial residual
    auto r = share(gko::read<vec>(std::ifstream("../data/rhs_2592000.mtx"), exec));
    auto one = gko::initialize<vec>({1.0}, exec);
    auto neg_one = gko::initialize<vec>({-1.0}, exec);
    auto initres = gko::initialize<vec>({0.0}, exec);
    A->apply(one, x, neg_one, r); // r=Ax-r (with r=b)
    r->compute_norm2(initres);

    // ---- Prepare the stopping criteria ----

    // Max iter
    auto iter_stop = gko::share(gko::stop::Iteration::build().with_max_iters(100u).on(exec));
    // relative tolerance criteria
    const gko::remove_complex<_TYPE_> tolerance = 5e-4;
    auto tol_stop = gko::share(gko::stop::ResidualNorm<_TYPE_>::build()
                                   .with_baseline(gko::stop::mode::initial_resnorm)
                                   .with_reduction_factor(tolerance)
                                   .on(exec));
    // Exact sol stop for coarse level
    auto exact_tol_stop = gko::share(gko::stop::ResidualNorm<_TYPE_>::build()
                                         .with_baseline(gko::stop::mode::rhs_norm)
                                         .with_reduction_factor(1e-14)
                                         .on(exec));

    // Add criteria to logger
    std::shared_ptr<const gko::log::Convergence<_TYPE_>> logger = gko::log::Convergence<_TYPE_>::create();
    iter_stop->add_logger(logger);
    tol_stop->add_logger(logger);

    // ---- customize some settings of the multigrid preconditioner. ----

    /*auto incomplete_cholesky_gen = gko::share(
        incomplete_cholesky_preconditioner::build()
            .with_factorization(incomplete_cholesky_factorisation::build())
            .on(exec));*/

    auto jacobi_gen = gko::share(
        jacobi_preconditioner::build()
            .with_max_block_size(max_block_size_jacobi)
            .on(exec));

    if (use_storage_optim_jacobi)
    {
        jacobi_gen = gko::share(
            jacobi_preconditioner::build()
                .with_max_block_size(max_block_size_jacobi)
                .with_storage_optimization(gko::precision_reduction::autodetect())
                .on(exec));
    }
    auto preconditioner = jacobi_gen;

    // Smoother
    auto smoother_gen = gko::share(gko::solver::build_smoother(preconditioner, n_smooth, relax_smooth));

    // Prologation/refinment algorithm
    auto multigrid_level_gen = gko::share(parallel_graph_matching::build().with_deterministic(pgm_deterministic).on(exec));

    // Next we select a CG solver for the coarsest level. Again, since the input
    // matrix is known to be spd, and the Pgm restriction preserves this
    // characteristic, we can safely choose the CG. We reuse the Ic factory here
    // to generate an Ic preconditioner. It is important to solve until machine
    // precision here to get a good convergence rate.
    auto coarsest_gen = gko::share(conjugate_gradient::build()
                                       .with_preconditioner(preconditioner)
                                       .with_criteria(iter_stop, exact_tol_stop)
                                       .on(exec));

    std::cout << "Creating multigrid factory...\n";
    // Create multigrid factory
    std::shared_ptr<gko::LinOpFactory> multigrid_gen;
    multigrid_gen =
        multigrid::build()
            // Amounts of subgrids
            .with_max_levels(max_levels)
            // Min size of the coarse grid
            //.with_min_coarse_rows(32u)
            .with_pre_smoother(smoother_gen)
            // Post smoother = pre smoother
            .with_post_uses_pre(true)
            // Algo to prolongation / refine grid
            .with_mg_level(multigrid_level_gen)
            // Solver for coarsest solver
            .with_coarsest_solver(coarsest_gen)
            .with_default_initial_guess(gko::solver::initial_guess_mode::zero)
            // Amount of iteration for the amg solver (1 as its used as a preconditioner)
            .with_criteria(gko::stop::Iteration::build().with_max_iters(n_v_cycles))
            .on(exec);

    // Create solver factory
    auto solver_gen = conjugate_gradient::build()
                          .with_criteria(iter_stop, tol_stop)
                          .with_preconditioner(multigrid_gen)
                          .on(exec);

    std::cout << "Creating solver ...\n";
    // Create solver
    std::chrono::nanoseconds gen_time(0);
    auto gen_tic = std::chrono::steady_clock::now();
    auto solver = solver_gen->generate(A);
    exec->synchronize();
    auto gen_toc = std::chrono::steady_clock::now();
    gen_time += std::chrono::duration_cast<std::chrono::nanoseconds>(gen_toc - gen_tic);

    std::cout << "\nAll ready... Solving !\n";

    // Solve system
    exec->synchronize();
    std::chrono::nanoseconds time(0);
    auto tic = std::chrono::steady_clock::now();
    mynvtxRangePush("Solving");
    solver->apply(b, x);
    exec->synchronize();
    mynvtxRangePop("Solving");
    auto toc = std::chrono::steady_clock::now();
    time += std::chrono::duration_cast<std::chrono::nanoseconds>(toc - tic);
    std::cout << "OK! results:\n\n";

    // Calculate residual
    auto res = gko::as<vec>(logger->get_residual_norm());

    std::cout << "Initial residual norm sqrt(r^T r): \n";
    write(std::cout, initres);
    std::cout << "Final residual norm sqrt(r^T r): \n";
    write(std::cout, res);

    // Print solver statistics
    std::cout << "CG iteration count:     " << logger->get_num_iterations()
              << std::endl;
    std::cout << "CG generation time [s]: "
              << static_cast<double>(gen_time.count()) / 1e9 << std::endl;
    std::cout << "CG execution time [s]: "
              << static_cast<double>(time.count()) / 1e9 << std::endl;
    std::cout << "CG execution time per iteration[ms]: "
              << static_cast<double>(time.count()) / 1e6 /
                     logger->get_num_iterations()
              << std::endl;
}
