/******************************************************************************
 *  Minimal serial example of reading a .mtx matrix and solving Ax = b using
 *  HYPRE with PCG + BoomerAMG.
 *
 *  To compile (adjust paths to match your environment):
 *    mpicc -I/path/to/hypre/include -L/path/to/hypre/lib -lhypre -o solve_pcg_amg solve_pcg_amg.c
 * 
 *  Run (in serial):
 *    mpirun -n 1 ./solve_pcg_amg matrix.mtx
 ******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

#include "HYPRE.h"
#include "HYPRE_parcsr_ls.h"
#include "HYPRE_utilities.h"
#include "_hypre_utilities.h"

/*--------------------------------------------------------------------------
 * A simple Matrix Market (.mtx) reader for a general sparse matrix in COO form.
 * This function reads:
 *   - number of rows (M), number of columns (N), and number of nonzeros (NNZ)
 *   - the triplets (i, j, val)
 * and returns them in arrays (rowIdx, colIdx, val).
 * 
 * Important: 
 *   - This example does NOT handle symmetric formats or special storage.
 *   - Rows and columns are zero-based internally for HYPRE, while some .mtx
 *     files may be 1-based.
 *--------------------------------------------------------------------------
 */

static int readMtxFile(
    const char *filename,
    int       *M,       /* out: number of rows    */
    int       *N,       /* out: number of cols    */
    int       *NNZ,     /* out: number of nonzeros*/
    int      **rowIdx,  /* out: row indices array */
    int      **colIdx,  /* out: col indices array */
    double   **values)  /* out: values array      */
{
    FILE *fp = fopen(filename, "r");
    if (!fp) {
        fprintf(stderr, "Error: cannot open file %s\n", filename);
        return -1;
    }

    // Skip comments
    char line[256];
    do {
        if (!fgets(line, sizeof(line), fp)) {
            fprintf(stderr, "Error: failed reading Matrix Market file header\n");
            fclose(fp);
            return -1;
        }
    } while (line[0] == '%');

    // Read matrix size and number of nonzeros
    int m, n, nnz;
    if (sscanf(line, "%d %d %d", &m, &n, &nnz) != 3) {
        fprintf(stderr, "Error: invalid Matrix Market header (size)\n");
        fclose(fp);
        return -1;
    }

    *M   = m;
    *N   = n;
    *NNZ = nnz;

    // Allocate arrays
    *rowIdx = (int*)    calloc(nnz, sizeof(int));
    *colIdx = (int*)    calloc(nnz, sizeof(int));
    *values = (double*) calloc(nnz, sizeof(double));

    // Read triplets
    for (int i = 0; i < nnz; i++) {
        int    ii, jj;
        double val;
        if (fscanf(fp, "%d %d %lf", &ii, &jj, &val) != 3) {
            fprintf(stderr, "Error: invalid read of triplet data at line %d\n", i + 1);
            fclose(fp);
            return -1;
        }
        // Convert 1-based (Matrix Market) to 0-based for internal use
        ii--; 
        jj--;

        (*rowIdx)[i] = ii;
        (*colIdx)[i] = jj;
        (*values)[i] = val;
    }

    fclose(fp);
    return 0;
}

int main(int argc, char *argv[])
{
    if (argc < 2) {
        fprintf(stderr, "Usage: %s matrix.mtx\n", argv[0]);
        return 1;
    }

    /* Initialize MPI */
    MPI_Init(&argc, &argv);

    /* Initialize HYPRE (optional in newer versions, but good practice) */
    HYPRE_Initialize();

    /* Read matrix from .mtx file */
    int M, N, NNZ;
    int    *hRowIdx = NULL;
    int    *hColIdx = NULL;
    double *hVals   = NULL;

    if (readMtxFile(argv[1], &M, &N, &NNZ, &hRowIdx, &hColIdx, &hVals) != 0) {
        MPI_Finalize();
        return 1;
    }

    // For this example, assume M == N (square matrix).
    if (M != N) {
        fprintf(stderr, "Error: matrix must be square!\n");
        MPI_Finalize();
        return 1;
    }

    // Number of rows in the global system
    HYPRE_Int globalNumRows = M;

    // In serial, our local range is the entire matrix
    HYPRE_Int ilower = 0;
    HYPRE_Int iupper = globalNumRows - 1;

    /*************************************************************************
     * 1) Create and populate the HYPRE_IJMatrix in serial
     *************************************************************************/
    HYPRE_IJMatrix ij_A;
    HYPRE_IJMatrixCreate(MPI_COMM_WORLD, ilower, iupper, ilower, iupper, &ij_A);

    // Set the object type to ParCSR (parallel CSR) - even in serial
    HYPRE_IJMatrixSetObjectType(ij_A, HYPRE_PARCSR);

    // (Optional) preallocate if you have an estimate of nonzeros/row
    // HYPRE_IJMatrixSetMaxOffProcElmts(ij_A, ...);

    // Initialize the IJMatrix
    HYPRE_IJMatrixInitialize(ij_A);

    // Insert the entries from the arrays
    // for (int i = 0; i < NNZ; i++) {
    //     HYPRE_Int row = (HYPRE_Int) hRowIdx[i];
    //     HYPRE_Int col = (HYPRE_Int) hColIdx[i];
    //     double     val = hVals[i];
    //     HYPRE_IJMatrixAddToValues(ij_A, 1, &col, &row, &val);
    // }

    for (int i = 0; i < NNZ; i++)
{
    HYPRE_Int    nrows    = 1;
    HYPRE_Int    ncols    = 1;
    HYPRE_BigInt rowIndex = (HYPRE_BigInt) hRowIdx[i];
    HYPRE_BigInt colIndex = (HYPRE_BigInt) hColIdx[i];
    double       val      = hVals[i];

    HYPRE_IJMatrixAddToValues(ij_A, 
                              nrows, &ncols, 
                              &rowIndex, 
                              &colIndex, 
                              &val);
}

    // Assemble the matrix
    HYPRE_IJMatrixAssemble(ij_A);

    // Extract the ParCSR object for the solver
    HYPRE_ParCSRMatrix A;
    HYPRE_IJMatrixGetObject(ij_A, (void**) &A);

    /*************************************************************************
     * 2) Create right-hand side (b) and solution (x) vectors
     *    Here we simply set b = 1.0 for all entries, and x = 0.0
     *************************************************************************/
    HYPRE_IJVector ij_b;
    HYPRE_IJVector ij_x;

    // Create
    HYPRE_IJVectorCreate(MPI_COMM_WORLD, ilower, iupper, &ij_b);
    HYPRE_IJVectorCreate(MPI_COMM_WORLD, ilower, iupper, &ij_x);

    // Set object type
    HYPRE_IJVectorSetObjectType(ij_b, HYPRE_PARCSR);
    HYPRE_IJVectorSetObjectType(ij_x, HYPRE_PARCSR);

    // Initialize
    HYPRE_IJVectorInitialize(ij_b);
    HYPRE_IJVectorInitialize(ij_x);

    // Set b = 1.0, x = 0.0
    double *rhsVals = (double*) calloc(globalNumRows, sizeof(double));
    double *xVals   = (double*) calloc(globalNumRows, sizeof(double));

    for (int i = 0; i < globalNumRows; i++) {
        rhsVals[i] = 1.0;  // b_i
        xVals[i]   = 0.0;  // initial guess
    }

    HYPRE_IJVectorSetValues(ij_b, globalNumRows, NULL, rhsVals);
    HYPRE_IJVectorSetValues(ij_x, globalNumRows, NULL, xVals);

    // Assemble
    HYPRE_IJVectorAssemble(ij_b);
    HYPRE_IJVectorAssemble(ij_x);

    // Get ParCSR objects
    HYPRE_ParVector b;
    HYPRE_ParVector x;
    HYPRE_IJVectorGetObject(ij_b, (void**) &b);
    HYPRE_IJVectorGetObject(ij_x, (void**) &x);

    /*************************************************************************
     * 3) Set up the PCG solver with BoomerAMG as a preconditioner
     *************************************************************************/
    HYPRE_Solver solver, precond;

    // Create PCG solver
    HYPRE_ParCSRPCGCreate(MPI_COMM_WORLD, &solver);

    // Set some parameters
    HYPRE_ParCSRPCGSetTol(solver, 1e-7);         // convergence tolerance
    HYPRE_ParCSRPCGSetMaxIter(solver, 1000);     // max iterations
    HYPRE_ParCSRPCGSetTwoNorm(solver, 1);        // use the two-norm as the stopping criteria
    HYPRE_ParCSRPCGSetPrintLevel(solver, 2);     // print residual norm each iteration

    // Create BoomerAMG preconditioner
    HYPRE_BoomerAMGCreate(&precond);

    // Set some BoomerAMG parameters (defaults are fine for minimal example)
    // E.g.: HYPRE_BoomerAMGSetCoarsenType(precond, 6);
    //       HYPRE_BoomerAMGSetRelaxType(precond, 6);
    //       HYPRE_BoomerAMGSetNumSweeps(precond, 2);
    //       HYPRE_BoomerAMGSetStrongThreshold(precond, 0.25);
    // ...

    // Set the PCG's preconditioner to be BoomerAMG
    HYPRE_ParCSRPCGSetPrecond(
        solver,
        (HYPRE_PtrToParSolverFcn) HYPRE_BoomerAMGSolve,
        (HYPRE_PtrToParSolverFcn) HYPRE_BoomerAMGSetup,
        precond);

    /*************************************************************************
     * 4) Solve the system
     *************************************************************************/
    HYPRE_ParCSRPCGSetup(solver, A, b, x);
    HYPRE_ParCSRPCGSolve(solver, A, b, x);

    // Get some info
    int    numIterations;
    double finalResidualNorm;
    HYPRE_ParCSRPCGGetNumIterations(solver, &numIterations);
    HYPRE_ParCSRPCGGetFinalRelativeResidualNorm(solver, &finalResidualNorm);

    if (0 == 0) { // rank == 0 in single-process, just print
        printf("\n");
        printf("PCG converged after %d iterations.\n", numIterations);
        printf("Final residual norm = %e\n", finalResidualNorm);
        printf("\n");
    }

    /*************************************************************************
     * 5) Cleanup
     *************************************************************************/
    free(rhsVals);
    free(xVals);
    free(hRowIdx);
    free(hColIdx);
    free(hVals);

    // Destroy solver and preconditioner
    HYPRE_ParCSRPCGDestroy(solver);
    HYPRE_BoomerAMGDestroy(precond);

    // Destroy matrix and vectors
    HYPRE_IJMatrixDestroy(ij_A);
    HYPRE_IJVectorDestroy(ij_b);
    HYPRE_IJVectorDestroy(ij_x);

    // Finalize HYPRE and MPI
    HYPRE_Finalize();
    MPI_Finalize();

    return 0;
}