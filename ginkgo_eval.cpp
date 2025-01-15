// This file is heavily inspired  the ginkgo's project examples:
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

using conjugate_gradient = gko::solver::Cg<_TYPE_>; //14 iters
using conjugate_gradient_square = gko::solver::Cgs<_TYPE_>; //7 iters
using biconjugate_gradient = gko::solver::Bicg<_TYPE_>; //Does not support AMG precond
using biconjugate_gradient_stabilized = gko::solver::Bicgstab<_TYPE_>; //7 iters
using flexible_conjugate_gradient = gko::solver::Fcg<_TYPE_>; //14 iters
using generalized_conjugate_residual =  gko::solver::Gcr<_TYPE_>; //13 iters
using multigrid = gko::solver::Multigrid;

using main_solver=conjugate_gradient;
using coarse_solver=conjugate_gradient;

using parallel_graph_matching = gko::multigrid::Pgm<_TYPE_, _STYPE_>;

using jacobi_preconditioner = gko::preconditioner::Jacobi<_TYPE_, _STYPE_>;
using gauss_seidel_preconditioner = gko::preconditioner::GaussSeidel<_TYPE_, _STYPE_>;
using incomplete_cholesky_preconditioner = gko::preconditioner::Ic<gko::solver::LowerTrs<_TYPE_>>;
using incomplete_cholesky_factorisation = gko::factorization::Ic<_TYPE_, _STYPE_>;

// Useful struct
struct ResolutionParams
{
    // Jacobi parameters
    int max_block_size_jacobi;
    bool use_storage_optim_jacobi;

    // Smoother parameters
    int n_smooth;
    double relax_smooth;

    // PGM parameters
    bool pgm_deterministic;

    // Multi-level cycle parameters
    int max_levels;
    int max_iter_amg_precond;
};

using A_t = std::shared_ptr<mtx>;
using b_t = std::shared_ptr<vec>;
using x_t = std::unique_ptr<vec>;
using exec_t = std::shared_ptr<gko::Executor>;
void evaluate_solver(const exec_t &exec, const A_t &A, const b_t &b, const x_t &x, const ResolutionParams &params, bool verbose = true, bool write_perfs = false)
{
    // Unpack into local variables:
    auto max_block_size_jacobi = params.max_block_size_jacobi;
    auto use_storage_optim_jacobi = params.use_storage_optim_jacobi;
    auto n_smooth = params.n_smooth;
    auto relax_smooth = params.relax_smooth;
    auto pgm_deterministic = params.pgm_deterministic;
    auto max_levels = params.max_levels;
    auto max_iter_amg_precond = params.max_iter_amg_precond;

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

    /*
    auto incomplete_cholesky_gen = gko::share(
        incomplete_cholesky_preconditioner::build()
            .with_factorization(incomplete_cholesky_factorisation::build())
            .on(exec));*/

    auto jacobi_gen = gko::share(
        jacobi_preconditioner::build()
//            .with_max_block_size(max_block_size_jacobi)
            .on(exec));

    /*
    auto gauss_seidel_gen = gko::share(
        gauss_seidel_preconditioner::build()
            .on(exec));*/

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
    // characteristic, we can safely choose the CG. We reuse the jacobi factory here
    // to generate an jacobi preconditioner. It is important to solve until machine
    // precision here to get a good convergence rate.
    /*auto coarsest_gen = gko::share(coarse_solver::build()
                                       .with_preconditioner(preconditioner)
                                       .with_criteria(iter_stop, exact_tol_stop)
                                       .on(exec));*/

    if (verbose) std::cout << "Creating multigrid factory...\n";
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
            // Solver for coarsest solver --> No solver
           // .with_coarsest_solver(coarsest_gen)
            .with_default_initial_guess(gko::solver::initial_guess_mode::zero)
            // Amount of iteration for the amg solver (1 as its used as a preconditioner)
            .with_criteria(gko::stop::Iteration::build().with_max_iters(max_iter_amg_precond))
            .on(exec);

    // Create solver factory
    auto solver_gen = main_solver::build()
                          .with_criteria(iter_stop, tol_stop)
                          .with_preconditioner(multigrid_gen)
                          .on(exec);

    if (verbose)
        std::cout << "Creating solver ...\n";

    // Create solver
    std::chrono::nanoseconds gen_time(0);
    auto gen_tic = std::chrono::steady_clock::now();
    auto solver = solver_gen->generate(A);
    exec->synchronize();
    auto gen_toc = std::chrono::steady_clock::now();
    gen_time += std::chrono::duration_cast<std::chrono::nanoseconds>(gen_toc - gen_tic);

    if (verbose)
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
    if (verbose)
        std::cout << "OK! results:\n\n";

    // Calculate residual
    auto res = gko::as<vec>(logger->get_residual_norm());
    if (verbose){
        std::cout << "Final residual norm sqrt(r^T r): \n";
        write(std::cout, res);}

    const int iteration_count = logger->get_num_iterations();
    const double generation_time = static_cast<double>(gen_time.count()) / 1e9;
    const double execution_time = static_cast<double>(time.count()) / 1e9;
    const double execution_time_per_iter = execution_time / 1e-3 / iteration_count;

    // Print solver statistics
    if (verbose)
        std::cout << "CG iteration count:     " << iteration_count
                  << std::endl;
    if (verbose)
        std::cout << "CG generation time [s]: "
                  << generation_time << std::endl;
    if (verbose)
        std::cout << "CG execution time [s]: "
                  << execution_time << std::endl;
    if (verbose)
        std::cout << "CG execution time per iteration[ms]: "
                  << execution_time_per_iter
                  << std::endl;

    if (write_perfs)
    {
        std::ofstream csv("perfs.csv", std::ios_base::app);
        csv << max_block_size_jacobi << ","
            << n_smooth << ", "
            << relax_smooth << ", "
            << max_levels << ", "
            << max_iter_amg_precond << ", "
            << iteration_count << ", "
            << generation_time << ", "
            << execution_time_per_iter << ", "
            << execution_time
            << "\n";
    }
    return;
}

int main(int argc, char *argv[])
{
    // ---- Read user input and select device ----

    // Print version information
    std::cout << "\n"
              << gko::version_info::get() << std::endl;

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
    std::cout << "Initial residual norm sqrt(r^T r): \n";
    write(std::cout, initres);

    // One resolution :
    
    // Resolution parameters
    ResolutionParams params{
        4u,   // max_block_size_jacobi
        true, // use_storage_optim_jacobi
        1u,   // n_smooth
        0.9,  // relax_smooth
        false, // pgm_deterministic
        50u,   // max_levels
        1u    // max_iter_amg_precond
    };

    evaluate_solver(exec, A, b, x, params);   

    /*
    // Ranges for each parameter:
    std::vector<int> block_sizes = {2, 4, 8};   // max_block_size_jacobi
    std::vector<int> smooth_values = {1, 2, 3};   // n_smooth
    std::vector<double> relax_values = { 0.8, 0.9};      // relax_smooth
    std::vector<int> level_values = {6, 8, 10};   // max_levels
    std::vector<int> v_cycles_values = {1, 2, 3}; // max_iter_amg_precond

    // For simplicity, we'll fix these to `true`. Adjust as needed.
    bool use_storage_optim_jacobi = true;
    bool pgm_deterministic = true;

    //mCompute the total number of runs (cartesian product of all param ranges).
    std::size_t total_runs = block_sizes.size() 
                           * smooth_values.size() 
                           * relax_values.size() 
                           * level_values.size() 
                           * v_cycles_values.size();

    //Create a counter to track progress.
    std::size_t counter = 0;

    // Nested loops to cover the entire grid:
    for (auto bs : block_sizes)
    {
        for (auto ns : smooth_values)
        {
            for (auto rs : relax_values)
            {
                for (auto ml : level_values)
                {
                    for (auto nvc : v_cycles_values)
                    {
                        // Build params for this specific combination
                        ResolutionParams params{
                            bs,                       // max_block_size_jacobi
                            use_storage_optim_jacobi, // use_storage_optim_jacobi
                            ns,                       // n_smooth
                            rs,                       // relax_smooth
                            pgm_deterministic,        // pgm_deterministic
                            ml,                       // max_levels
                            nvc                       // max_iter_amg_precond
                        };

                        // Launch your solver with these parameters
                        evaluate_solver(exec, A, b, x, params, false, true);
                        x->copy_from(host_x);//reset x for fair comparison
                        counter+=1;
                        std::cout<<"Progession: "<<counter<<" / "<<total_runs<<std::endl;
                    }
                }
            }
        }
    }
    */
    
}
