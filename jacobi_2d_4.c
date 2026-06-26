#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "mpi.h"

typedef struct {
    int row;
    int col;
    double charge;
} Charge;

#define NUM_CHARGES 6

Charge charges[NUM_CHARGES] = {
    {100, 50, 2.0},
    {100, 150, 2.0},
    {50,  30,  -1.0},
    {150, 30,  -1.0},
    {50, 130,  -1.0},
    {150, 130, -1.0}
};

// Function prototypes
double **initialize_grid_tobo(int rows, int cols, float scale); //"tobo" = top-bottom
double **initialize_grid_zero(int rows, int cols);
double **initialize_grid_step(int rows, int cols, float scale);
double **initialize_grid_grad(int rows, int cols, float scale);
double **initialize_grid_grad_hor(int rows, int cols, float scale);

void free_grid(double **grid, int rows);
int save_grid_as_matrix(double **grid, int rows, int cols, const char *filename, int iteration);
void factor_processes(int n, int rows, int cols, int *px, int *py);

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int numP, myID;
    MPI_Comm_size(MPI_COMM_WORLD, &numP);
    MPI_Comm_rank(MPI_COMM_WORLD, &myID);

    if (argc < 4) {
        if (myID == 0) {
            printf("Usage: %s rows cols error_threshold\n", argv[0]);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    int rows = atoi(argv[1]);
    int cols = atoi(argv[2]);
    float errThres = (float)atof(argv[3]);

    if (rows < 3 || cols < 3) {
        if (myID == 0) {
            printf("Rows and cols must be at least 3.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    int px, py;
    factor_processes(numP, rows, cols, &px, &py);
    if (px == -1 || py == -1) {
        if (myID == 0) {
            printf("Cannot factor %d processes with grid %dx%d.\n", numP, rows, cols);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    int myRows = rows / px;
    int myCols = cols / py;
    int stride = myCols + 2;   // total columns including left/right ghosts

    if (myID == 0){
        printf("Process divided in blocks of rows = %d and columns = %d\n",myRows,myCols);
    }
    double **data = NULL;
    double *flat_data = NULL;
    double *rho_flat = NULL;

    // Root initialises the grid and rho
    if (myID == 0) {
        data = initialize_grid_grad(rows, cols, 3.0);
        if (!data) {
            printf("Failed to allocate grid.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        rho_flat = (double *)calloc(rows * cols, sizeof(double));
        for (int c = 0; c < NUM_CHARGES; c++) {
            int gr = charges[c].row;
            int gc = charges[c].col;
            rho_flat[gr * cols + gc] = charges[c].charge;
            rho_flat[(gr + 1) * cols + gc] = charges[c].charge;
            rho_flat[(gr + 1) * cols + gc + 1] = charges[c].charge;
            rho_flat[(gr + 1) * cols + gc - 1] = charges[c].charge;
            rho_flat[(gr - 1) * cols + gc] = charges[c].charge;
            rho_flat[(gr - 1) * cols + gc + 1] = charges[c].charge;
            rho_flat[(gr - 1) * cols + gc - 1] = charges[c].charge;
            rho_flat[gr * cols + gc + 1] = charges[c].charge;
            rho_flat[gr * cols + gc - 1] = charges[c].charge;
        }

        // Save initial matrix (data + rho)
        double **combined = (double **)malloc(rows * sizeof(double *));
        for (int i = 0; i < rows; i++) {
            combined[i] = (double *)malloc(cols * sizeof(double));
            for (int j = 0; j < cols; j++) {
                combined[i][j] = data[i][j] + rho_flat[i * cols + j];
            }
        }
        save_grid_as_matrix(combined, rows, cols, "initial_matrix.csv", -1);
        for (int i = 0; i < rows; i++) free(combined[i]);
        free(combined);

        // Flatten grid
        flat_data = (double *)malloc(rows * cols * sizeof(double));
        for (int i = 0; i < rows; i++) {
            memcpy(&flat_data[i * cols], data[i], cols * sizeof(double));
        }
    }

    // --- Create derived datatypes (must be before scatter) ---
    MPI_Datatype row_type, col_type, block_type, block_type_resized;
    MPI_Type_contiguous(myCols, MPI_DOUBLE, &row_type);
    MPI_Type_vector(myRows, 1, stride, MPI_DOUBLE, &col_type);
    MPI_Type_vector(myRows, myCols, stride, MPI_DOUBLE, &block_type);
    MPI_Type_create_resized(block_type, 0, myRows * myCols * sizeof(double), &block_type_resized);
    MPI_Type_commit(&row_type);
    MPI_Type_commit(&col_type);
    MPI_Type_commit(&block_type_resized);
    MPI_Type_free(&block_type);

    // --- Pack and scatter ---
    double *packed_grid = NULL;
    double *packed_rho = NULL;
    if (myID == 0) {
        packed_grid = (double *)malloc(numP * myRows * myCols * sizeof(double));
        packed_rho  = (double *)malloc(numP * myRows * myCols * sizeof(double));
        for (int p = 0; p < numP; p++) {
            int pr = p / py;
            int pc = p % py;
            int start_row = pr * myRows;
            int start_col = pc * myCols;
            for (int i = 0; i < myRows; i++) {
                memcpy(&packed_grid[p * myRows * myCols + i * myCols],
                       &flat_data[(start_row + i) * cols + start_col],
                       myCols * sizeof(double));
                memcpy(&packed_rho[p * myRows * myCols + i * myCols],
                       &rho_flat[(start_row + i) * cols + start_col],
                       myCols * sizeof(double));
            }
        }
        free(rho_flat);   // no longer needed on root
        // flat_data is kept for final gather
    }

    // Allocate local arrays with ghosts
    double *myData = (double *)malloc((myRows + 2) * stride * sizeof(double));
    double *buff   = (double *)malloc((myRows + 2) * stride * sizeof(double));
    double *rho_local = (double *)malloc(myRows * myCols * sizeof(double));

    // Scatter grid using block_type_resized (handles stride)
    MPI_Scatter(packed_grid, myRows * myCols, MPI_DOUBLE,
                &myData[1 * stride + 1], 1, block_type_resized,
                0, MPI_COMM_WORLD);

    // Scatter rho (contiguous buffer)
    MPI_Scatter(packed_rho, myRows * myCols, MPI_DOUBLE,
                rho_local, myRows * myCols, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    if (myID == 0) {
        free(packed_grid);
        free(packed_rho);
        // flat_data will be used for gather
    }

    // Initialise ghost cells (they will be overwritten in first exchange)
    for (int i = 0; i < myRows + 2; i++) {
        myData[i * stride + 0] = 0.0;
        myData[i * stride + myCols + 1] = 0.0;
    }
    for (int j = 0; j < myCols + 2; j++) {
        myData[0 * stride + j] = 0.0;
        myData[(myRows + 1) * stride + j] = 0.0;
    }

    // Copy initial guess to buff
    memcpy(buff, myData, (myRows + 2) * stride * sizeof(double));

    // Determine neighbours in process grid
    int my_px = myID / py;
    int my_py = myID % py;
    int up    = (my_px > 0) ? (my_px - 1) * py + my_py : MPI_PROC_NULL;
    int down  = (my_px < px - 1) ? (my_px + 1) * py + my_py : MPI_PROC_NULL;
    int left  = (my_py > 0) ? my_px * py + (my_py - 1) : MPI_PROC_NULL;
    int right = (my_py < py - 1) ? my_px * py + (my_py + 1) : MPI_PROC_NULL;

    // Start timing
    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();

    float error = errThres + 1.0f;
    float myError = 0.0f;
    MPI_Request reqs[8];
    int idx;

    while (error > errThres) {
    myError = 0.0f;

    // ---- 1. Initiate non‑blocking exchanges ----
    idx = 0;
    if (up != MPI_PROC_NULL)
        MPI_Irecv(&myData[0 * stride + 1], 1, row_type, up, 0, MPI_COMM_WORLD, &reqs[idx++]);
    if (down != MPI_PROC_NULL)
        MPI_Irecv(&myData[(myRows + 1) * stride + 1], 1, row_type, down, 1, MPI_COMM_WORLD, &reqs[idx++]);
    if (left != MPI_PROC_NULL)
        MPI_Irecv(&myData[1 * stride + 0], 1, col_type, left, 2, MPI_COMM_WORLD, &reqs[idx++]);
    if (right != MPI_PROC_NULL)
        MPI_Irecv(&myData[1 * stride + (myCols + 1)], 1, col_type, right, 3, MPI_COMM_WORLD, &reqs[idx++]);

    if (up != MPI_PROC_NULL)
        MPI_Isend(&myData[1 * stride + 1], 1, row_type, up, 1, MPI_COMM_WORLD, &reqs[idx++]);
    if (down != MPI_PROC_NULL)
        MPI_Isend(&myData[myRows * stride + 1], 1, row_type, down, 0, MPI_COMM_WORLD, &reqs[idx++]);
    if (left != MPI_PROC_NULL)
        MPI_Isend(&myData[1 * stride + 1], 1, col_type, left, 3, MPI_COMM_WORLD, &reqs[idx++]);
    if (right != MPI_PROC_NULL)
        MPI_Isend(&myData[1 * stride + myCols], 1, col_type, right, 2, MPI_COMM_WORLD, &reqs[idx++]);

    // ---- 2. Compute interior (no ghost dependencies) ----
    // Only if there is at least one interior row/col
    if (myRows >= 3 && myCols >= 3) {
        for (int i = 2; i <= myRows - 1; i++) {
            for (int j = 2; j <= myCols - 1; j++) {
                double new_val = 0.25 * (myData[(i - 1) * stride + j] +
                                         myData[(i + 1) * stride + j] +
                                         myData[i * stride + j - 1] +
                                         myData[i * stride + j + 1])
                                 - 0.25 * rho_local[(i - 1) * myCols + (j - 1)];
                buff[i * stride + j] = new_val;
                double diff = fabs(new_val - myData[i * stride + j]);
                if (diff > myError) myError = (float)diff;
            }
        }
    }

    // ---- 3. Wait for ghosts ----
    MPI_Waitall(idx, reqs, MPI_STATUSES_IGNORE);

    // ---- 4. Compute boundary rows/cols (now ghosts are valid) ----
    for (int i = 1; i <= myRows; i++) {
        for (int j = 1; j <= myCols; j++) {
            // Skip interior points (already computed)
            if (i >= 2 && i <= myRows - 1 && j >= 2 && j <= myCols - 1)
                continue;

            // Check if this is a physical boundary (should not be updated)
            int is_boundary = 0;
            if (my_px == 0 && i == 1) is_boundary = 1;
            if (my_px == px - 1 && i == myRows) is_boundary = 1;
            if (my_py == 0 && j == 1) is_boundary = 1;
            if (my_py == py - 1 && j == myCols) is_boundary = 1;

            if (!is_boundary) {
                double new_val = 0.25 * (myData[(i - 1) * stride + j] +
                                         myData[(i + 1) * stride + j] +
                                         myData[i * stride + j - 1] +
                                         myData[i * stride + j + 1])
                                 - 0.25 * rho_local[(i - 1) * myCols + (j - 1)];
                buff[i * stride + j] = new_val;
                double diff = fabs(new_val - myData[i * stride + j]);
                if (diff > myError) myError = (float)diff;
            } else {
                // Keep unchanged (copy old value)
                buff[i * stride + j] = myData[i * stride + j];
            }
        }
    }

    // ---- Swap buffers ----
    double *temp = myData;
    myData = buff;
    buff = temp;

    MPI_Allreduce(&myError, &error, 1, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);
}

    double end_time = MPI_Wtime();

    // --- Gather results ---
// Pack each process's interior into a contiguous send buffer
double *send_buf = (double *)malloc(myRows * myCols * sizeof(double));
for (int i = 0; i < myRows; i++) {
    memcpy(&send_buf[i * myCols], &myData[(i + 1) * stride + 1], myCols * sizeof(double));
}

double *recv_buffer = NULL;
int *recv_counts = NULL, *displs = NULL;
if (myID == 0) {
    recv_buffer = (double *)malloc(numP * myRows * myCols * sizeof(double));
    recv_counts = (int *)malloc(numP * sizeof(int));
    displs = (int *)malloc(numP * sizeof(int));
    for (int p = 0; p < numP; p++) {
        recv_counts[p] = myRows * myCols;
        displs[p] = p * myRows * myCols;
    }
}

MPI_Gatherv(send_buf, myRows * myCols, MPI_DOUBLE,
            recv_buffer, recv_counts, displs, MPI_DOUBLE,
            0, MPI_COMM_WORLD);

free(send_buf);

if (myID == 0) {
    // ----- UNPACK the packed buffer into flat_data (global layout) -----
    // flat_data is already allocated and has size rows * cols
    for (int p = 0; p < numP; p++) {
        int pr = p / py;          // row index in process grid
        int pc = p % py;          // col index
        int start_row = pr * myRows;
        int start_col = pc * myCols;
        // Copy block from recv_buffer into flat_data at the correct position
        for (int i = 0; i < myRows; i++) {
            memcpy(&flat_data[(start_row + i) * cols + start_col],
                   &recv_buffer[p * myRows * myCols + i * myCols],
                   myCols * sizeof(double));
        }
    }

    // Now flat_data contains the correctly assembled grid.
    double **final_grid = (double **)malloc(rows * sizeof(double *));
    for (int i = 0; i < rows; i++) {
        final_grid[i] = &flat_data[i * cols];
    }
    save_grid_as_matrix(final_grid, rows, cols, "final_grid_matrix.csv", -1);
    printf("Time with %d processes: %f seconds.\n", numP, end_time - start_time);
    
    FILE *logfp = fopen("2D_log.csv", "a");
	if (logfp) {
	    // Get current time for timestamp
	    time_t now;
	    time(&now);
	    struct tm *local = localtime(&now);
	    char timestamp[32];
	    strftime(timestamp, sizeof(timestamp), "%Y-%m-%d %H:%M:%S", local);

	    // Write one line: timestamp, rows, cols, error threshold, #processes, elapsed time
	    fprintf(logfp, "%s, %d, %d, %.6f, %d, %.6f\n",
		    timestamp, rows, cols, errThres, numP, end_time - start_time);
	    fclose(logfp);
	} else {
	    fprintf(stderr, "Warning: Could not open log file for writing.\n");
	}

    free(final_grid);
    free(recv_buffer);
    free(recv_counts);
    free(displs);
    free(flat_data);
    free_grid(data, rows);
}

    // Cleanup
    free(myData);
    free(buff);
    free(rho_local);
    MPI_Type_free(&row_type);
    MPI_Type_free(&col_type);
    MPI_Type_free(&block_type_resized);

    MPI_Finalize();
    return 0;
}

// ---- Utility functions (unchanged) ----
double **initialize_grid_tobo(int rows, int cols, float scale) {
    if (rows < 3 || cols < 3) {
        return NULL;
    }

    double **grid = (double **)malloc(rows * sizeof(double *));
    if (!grid) return NULL;

    for (int i = 0; i < rows; i++) {
        grid[i] = (double *)malloc(cols * sizeof(double));
        if (!grid[i]) {
            for (int k = 0; k < i; k++) free(grid[k]);
            free(grid);
            return NULL;
        }
        for (int j = 0; j < cols; j++) {
            grid[i][j] = 0.0;
        }
    }
    
    // Top row = scale
    for (int j = 0; j < cols; j++) {
        grid[0][j] = scale;
    }

    // Bottom row = -scale
    for (int j = 0; j < cols; j++) {
        grid[rows - 1][j] = -scale;
    }

    return grid;
}
double **initialize_grid_zero(int rows, int cols) {
    if (rows < 3 || cols < 3) {
        return NULL;
    }

    double **grid = (double **)malloc(rows * sizeof(double *));
    if (!grid) return NULL;

    for (int i = 0; i < rows; i++) {
        grid[i] = (double *)malloc(cols * sizeof(double));
        if (!grid[i]) {
            for (int k = 0; k < i; k++) free(grid[k]);
            free(grid);
            return NULL;
        }
        for (int j = 0; j < cols; j++) {
            grid[i][j] = 0.0;
        }
    }

    return grid;
}


double **initialize_grid_step(int rows, int cols, float scale) {
    if (rows < 3 || cols < 3) {
        return NULL;
    }

    double **grid = (double **)malloc(rows * sizeof(double *));
    if (!grid) return NULL;

    for (int i = 0; i < rows; i++) {
        grid[i] = (double *)malloc(cols * sizeof(double));
        if (!grid[i]) {
            for (int k = 0; k < i; k++) free(grid[k]);
            free(grid);
            return NULL;
        }
        for (int j = 0; j < cols; j++) {
            grid[i][j] = 0.0;
        }
    }
    
    // Top row = 1.0
    for (int j = 0; j < cols; j++) {
        grid[0][j] = scale;
    }

    // Bottom row = -1.0
    for (int j = 0; j < cols; j++) {
        grid[rows - 1][j] = -scale;
    }
    
    for (int i = 0; i < rows/2; i++) {
        grid[i][0] = scale;
        grid[i + rows/2][0] = -scale;
        
        grid[i][cols - 1] = scale;
        grid[i + rows/2][cols -1] = -scale;
    }

    return grid;
}

double **initialize_grid_grad(int rows, int cols, float scale) {
    if (rows < 3 || cols < 3) {
        return NULL;
    }

    double **grid = (double **)malloc(rows * sizeof(double *));
    if (!grid) return NULL;

    for (int i = 0; i < rows; i++) {
        grid[i] = (double *)malloc(cols * sizeof(double));
        if (!grid[i]) {
            for (int k = 0; k < i; k++) free(grid[k]);
            free(grid);
            return NULL;
        }
        for (int j = 0; j < cols; j++) {
            grid[i][j] = 0.0;
        }
    }
    
    // Top row = 1.0
    for (int j = 0; j < cols; j++) {
        grid[0][j] = scale;
    }

    // Bottom row = -1.0
    for (int j = 0; j < cols; j++) {
        grid[rows - 1][j] = -scale;
    }
    
    for (int i = 0; i < rows; i++) {
        grid[i][0] = scale*(1 - 2.0*(float)(i)/(float)rows);
        
        grid[i][cols - 1] = scale*(1 - 2.0*(float)(i)/(float)rows);
    }

    return grid;
}

double **initialize_grid_grad_hor(int rows, int cols, float scale) {
    if (rows < 3 || cols < 3) {
        return NULL;
    }

    double **grid = (double **)malloc(rows * sizeof(double *));
    if (!grid) return NULL;

    for (int i = 0; i < rows; i++) {
        grid[i] = (double *)malloc(cols * sizeof(double));
        if (!grid[i]) {
            for (int k = 0; k < i; k++) free(grid[k]);
            free(grid);
            return NULL;
        }
        for (int j = 0; j < cols; j++) {
            grid[i][j] = 0.0;
        }
    }
    
    // Top row = 1.0
    for (int j = 0; j < cols; j++) {
        grid[j][0] = scale;
    }

    // Bottom row = -1.0
    for (int j = 0; j < cols; j++) {
        grid[j][rows - 1] = -scale;
    }
    
    for (int i = 0; i < rows; i++) {
        grid[0][i] = scale*(1 - 2.0*(float)(i)/(float)rows);
        
        grid[cols - 1][i] = scale*(1 - 2.0*(float)(i)/(float)rows);
    }

    return grid;
}

void free_grid(double **grid, int rows) {
    if (!grid) return;
    for (int i = 0; i < rows; i++) free(grid[i]);
    free(grid);
}

int save_grid_as_matrix(double **grid, int rows, int cols, const char *filename, int iteration) {
    if (!grid || rows < 1 || cols < 1 || !filename) return 0;
    char full_filename[256];
    if (iteration >= 0)
        snprintf(full_filename, sizeof(full_filename), "grid_matrix_%04d.csv", iteration);
    else
        snprintf(full_filename, sizeof(full_filename), "%s", filename);
    FILE *fp = fopen(full_filename, "w");
    if (!fp) {
        fprintf(stderr, "Could not open %s for writing.\n", full_filename);
        return 0;
    }
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            fprintf(fp, "%.8f", grid[i][j]);
            if (j < cols - 1) fprintf(fp, ",");
        }
        fprintf(fp, "\n");
    }
    fclose(fp);
    return 1;
}

void factor_processes(int n, int rows, int cols, int *px, int *py) {
    int best_px = 1, best_py = n;
    int best_diff = n;
    for (int p = 1; p * p <= n; p++) {
        if (n % p == 0) {
            int q = n / p;
            if (rows % p == 0 && cols % q == 0) {
                int diff = abs(p - q);
                if (diff < best_diff) {
                    best_diff = diff;
                    best_px = p; best_py = q;
                }
            }
            if (rows % q == 0 && cols % p == 0) {
                int diff = abs(q - p);
                if (diff < best_diff) {
                    best_diff = diff;
                    best_px = q; best_py = p;
                }
            }
        }
    }
    if (best_diff == n) {
        *px = *py = -1;
    } else {
        *px = best_px;
        *py = best_py;
    }
}
