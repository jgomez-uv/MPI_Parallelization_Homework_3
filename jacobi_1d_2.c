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
    {100, 50, 2.0},    // O
    {100, 150, 2.0},   // O
    {50,  30,  -1.0},  // H
    {150, 30,  -1.0},  // H
    {50, 130,  -1.0},  // H
    {150, 130, -1.0}   // H
};

    // The charges are flipped in polarity as the real case, the real poisson eq for these (electrostatic eq) is d phi = -rho/epsilon, but our code works with the flipped d phi = rho ("d" being the corresponding lagrangian, and epsilon assumed to be an insignificant constant for numerical purposes).

double **initialize_grid_tobo(int rows, int cols, float scale); //"tobo" = top-bottom
double **initialize_grid_zero(int rows, int cols);
double **initialize_grid_step(int rows, int cols, float scale);
double **initialize_grid_grad(int rows, int cols, float scale);
void free_grid(double **grid, int rows);
int save_grid_as_matrix(double **grid, int rows, int cols, const char *filename, int iteration);

int main (int argc, char *argv[]){
    MPI_Init(&argc, &argv);
    
    int numP;
    MPI_Comm_size(MPI_COMM_WORLD, &numP);

    int myID;
    MPI_Comm_rank(MPI_COMM_WORLD, &myID);

    if(argc < 4){
        if(myID == 0){
            printf("The program must be called as: ./jacobi_1d rows cols errorThresh\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    int rows = atoi(argv[1]);
    int cols = atoi(argv[2]);
    float errThres = atof(argv[3]);

    if ((rows < 3) || (cols < 3)){
        if(myID == 0){
            printf("The row and col number must be above 2.\n");
            MPI_Abort(MPI_COMM_WORLD,1);
        }
    }

    if(rows % numP != 0){
        if(myID == 0){
            printf("The number of rows must be a multiple of number of processes.\n");
            MPI_Abort(MPI_COMM_WORLD,1);
        }
    }

    double **data = NULL;
    double *flat_data = NULL;
    double *rho = NULL;
    
    if(myID == 0){
        data = initialize_grid_grad(rows, cols, 3.0);
        rho = (double*)calloc(rows * cols, sizeof(double));
        for (int c = 0; c < NUM_CHARGES; c++) {
            int gr = charges[c].row;
            int gc = charges[c].col;
            // to get bigger charges, we add the charge to all vertical, horizontal and diagonal points.
            rho[gr * cols + gc] = charges[c].charge;
            rho[(gr+1) * cols + gc] = charges[c].charge;
            rho[(gr+1) * cols + gc + 1] = charges[c].charge;
            rho[(gr+1) * cols + gc - 1] = charges[c].charge;
            rho[(gr-1) * cols + gc] = charges[c].charge;
            rho[(gr-1) * cols + gc + 1] = charges[c].charge;
            rho[(gr-1) * cols + gc - 1] = charges[c].charge;
            rho[gr * cols + gc + 1] = charges[c].charge;
            rho[gr * cols + gc - 1] = charges[c].charge;
        }
        
        if (!data) {
            printf("Failed to allocate grid\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        
        double **combined = (double**)malloc(rows * sizeof(double*));
        for (int i = 0; i < rows; i++) {
            combined[i] = (double*)malloc(cols * sizeof(double));
            for (int j = 0; j < cols; j++) {
                combined[i][j] = data[i][j] + rho[i*cols + j];
        }
    }
    save_grid_as_matrix(combined, rows, cols, "initial_matrix.csv", -1);
    for (int i = 0; i < rows; i++) free(combined[i]);
    free(combined);
        
        // Flatten the grid for MPI communication
        flat_data = (double*)malloc(rows * cols * sizeof(double));
        for (int i = 0; i < rows; i++) {
            memcpy(&flat_data[i*cols], data[i], cols * sizeof(double));
        }
    }

    int myRows = rows / numP;

    MPI_Barrier(MPI_COMM_WORLD);

    double start = MPI_Wtime();

    // Allocate local data as double
    double *myData = (double*)malloc(myRows * cols * sizeof(double));
    double *buff = (double*)malloc(myRows * cols * sizeof(double));
    double *rho_local = (double*)malloc(myRows * cols * sizeof(double));
    
    // Scatter the data
    MPI_Scatter(flat_data, myRows * cols, MPI_DOUBLE, 
                myData, myRows * cols, MPI_DOUBLE, 
                0, MPI_COMM_WORLD);
    MPI_Scatter(rho, myRows * cols, MPI_DOUBLE,
            rho_local, myRows * cols, MPI_DOUBLE,
            0, MPI_COMM_WORLD);
    
    memcpy(buff, myData, myRows * cols * sizeof(double));
    
    if (myID == 0) {
        free(rho);
    }
    
    float error = errThres + 1.0;
    float myError;

    double *prevRow = (double*)malloc(cols * sizeof(double));
    double *nextRow = (double*)malloc(cols * sizeof(double));
    double *sendTop = (double*)malloc(cols * sizeof(double));
    double *sendBottom = (double*)malloc(cols * sizeof(double));
    
    MPI_Request request[4];
    MPI_Status status[4];

    while (error > errThres){
        myError = 0.0;
        
        // Exchange boundary rows
        if (myID > 0){
            memcpy(sendTop, myData, cols * sizeof(double));
            MPI_Isend(sendTop, cols, MPI_DOUBLE, myID-1, 0, MPI_COMM_WORLD, &request[0]);
            MPI_Irecv(prevRow, cols, MPI_DOUBLE, myID-1, 0, MPI_COMM_WORLD, &request[1]);
        }

        if (myID < numP-1){
            memcpy(sendBottom, &myData[(myRows-1)*cols], cols * sizeof(double));
            MPI_Isend(sendBottom, cols, MPI_DOUBLE, myID+1, 0, MPI_COMM_WORLD, &request[2]);
            MPI_Irecv(nextRow, cols, MPI_DOUBLE, myID+1, 0, MPI_COMM_WORLD, &request[3]);
        }

        // Update interior rows (not boundary rows of the local partition)
        for (int i = 1; i < myRows - 1; i++){
            for (int j = 1; j < cols - 1; j++){
                double new_val = 0.25 * (myData[(i+1)*cols+j] + 
                                         myData[i*cols+j-1] + 
                                         myData[i*cols+j+1] + 
                                         myData[(i-1)*cols+j])
                                         - 0.25 * rho_local[i*cols + j];
                buff[i*cols+j] = new_val;
                double diff = fabs(new_val - myData[i*cols+j]);
                if (diff > myError) myError = diff;
            }
        }

        // Update top boundary row of this process (if not the global top)
        if (myID > 0){
            MPI_Wait(&request[1], &status[1]);
            if (myRows >= 1){
                for (int j = 1; j < cols - 1; j++){
                    double new_val = 0.25 * (myData[cols+j] + 
                                             myData[j-1] + 
                                             myData[j+1] + 
                                             prevRow[j] - rho_local[j]);
                    buff[j] = new_val;
                    double diff = fabs(new_val - myData[j]);
                    if (diff > myError) myError = diff;
                }
            }
            MPI_Wait(&request[0], &status[0]);  // Wait for send to complete
        }

        // Update bottom boundary row of this process (if not the global bottom)
        if (myID < numP-1){
            MPI_Wait(&request[3], &status[3]);
            if (myRows >= 1){
                int last = (myRows-1) * cols;
                int prev = (myRows-2) * cols;
                for (int j = 1; j < cols - 1; j++){
                    double new_val = 0.25 * (nextRow[j] + 
                                             myData[last+j-1] + 
                                             myData[last+j+1] + 
                                             myData[prev+j] - rho_local[last + j]);
                    buff[last+j] = new_val;
                    double diff = fabs(new_val - myData[last+j]);
                    if (diff > myError) myError = diff;
                }
            }
            MPI_Wait(&request[2], &status[2]);  // Wait for send to complete
        }

        // Swap buffers
        double *temp = myData;
        myData = buff;
        buff = temp;

        // Global reduction for error
        MPI_Allreduce(&myError, &error, 1, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);
    }
    
    // Gather results
    MPI_Gather(myData, myRows * cols, MPI_DOUBLE, 
               flat_data, myRows * cols, MPI_DOUBLE, 
               0, MPI_COMM_WORLD);

    double end = MPI_Wtime();

    if (myID == 0){
        double **final_grid = (double**)malloc(rows * sizeof(double*));
        for (int i = 0; i < rows; i++) {
            final_grid[i] = &flat_data[i * cols];
        }
    
        save_grid_as_matrix(final_grid, rows, cols, "final_grid_matrix.csv", -1);
        
        printf("Time with %d processes: %f seconds.\n", numP, end-start);
        
        FILE *logfp = fopen("1D_log.csv", "a");
	if (logfp) {
	    // Get current time for timestamp
	    time_t now;
	    time(&now);
	    struct tm *local = localtime(&now);
	    char timestamp[32];
	    strftime(timestamp, sizeof(timestamp), "%Y-%m-%d %H:%M:%S", local);

	    // Write one line: timestamp, rows, cols, error threshold, #processes, elapsed time
	    fprintf(logfp, "%s, %d, %d, %.6f, %d, %.6f\n",
		    timestamp, rows, cols, errThres, numP, end - start);
	    fclose(logfp);
	} else {
	    fprintf(stderr, "Warning: Could not open log file for writing.\n");
	}
        
        free(final_grid);
        free(flat_data);
        free_grid(data, rows);
    }

    free(myData);
    free(buff);
    free(rho_local);
    free(prevRow);
    free(nextRow);
    free(sendTop);
    free(sendBottom);

    MPI_Finalize();
    return 0;
}

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


void free_grid(double **grid, int rows) {
    if (!grid) return;
    for (int i = 0; i < rows; i++) {
        free(grid[i]);
    }
    free(grid);
}

int save_grid_as_matrix(double **grid, int rows, int cols, const char *filename, int iteration) {
    if (!grid || rows < 1 || cols < 1 || !filename) {
        return 0;
    }
    
    char full_filename[256];
    if (iteration >= 0) {
        snprintf(full_filename, sizeof(full_filename), "grid_matrix_%04d.csv", iteration);
    } else {
        snprintf(full_filename, sizeof(full_filename), "%s", filename);
    }
    
    FILE *fp = fopen(full_filename, "w");
    if (!fp) {
        fprintf(stderr, "Error: Could not open file '%s' for writing\n", full_filename);
        return 0;
    }
    
    // Write as matrix (rows x cols)
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            fprintf(fp, "%.8f", grid[i][j]);
            if (j < cols - 1) {
                fprintf(fp, ",");
            }
        }
        fprintf(fp, "\n");
    }
    
    fclose(fp);
    return 1;
}
