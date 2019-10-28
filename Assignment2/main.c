#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <mpi.h>
#include "bitmap.h"

#define XSIZE 2560 // Size of before image
#define YSIZE 2048

void alter(uchar *image, int rank);

int block_size;

int main(void)
{
    // Initialize MPI Env and fetching env properties
    MPI_Init(NULL, NULL);
    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // Calculate the number of rows for each process and the last process (special case)
    block_size = (int)ceil(YSIZE / (double)world_size);
    int last_block_size = (int)YSIZE - (block_size * (world_size - 1));
    int block_sizes[world_size];
    int displacement[world_size];

    // Creating array with the block sizes for each process and the offset in the data
    for (int i = 0; i < world_size - 1; i++)
    {
        block_sizes[i] = block_size * 3 * XSIZE;
    }
    block_sizes[world_size - 1] = last_block_size * 3 * XSIZE;

    displacement[0] = 0;
    for (int i = 1; i < world_size; i++)
    {
        displacement[i] = displacement[i - 1] + block_sizes[i];
    }

    uchar *raw_image;

    // Main process load image
    if (world_rank == 0)
    {
        raw_image = malloc(XSIZE * YSIZE * 3);
        readbmp("before.bmp", raw_image);
    }

    uchar *image = malloc(block_sizes[world_rank] * 3 * XSIZE);

    // Send raw image data to the different proceeses, usign scatter so only needed data is sent to each process not all from the rank 0 process
    // Then altering the image on each process and ending with the processes gathering the data in the process rank 0
    MPI_Scatterv(raw_image, block_sizes, displacement, MPI_BYTE, image, block_sizes[world_rank], MPI_BYTE, 0, MPI_COMM_WORLD);
    alter(image, world_rank);
    MPI_Gatherv(image, block_sizes[world_rank], MPI_BYTE, raw_image, block_sizes, displacement, MPI_BYTE, 0, MPI_COMM_WORLD);

    // Saves the raw_image altered in the gather function and free up the space
    if (world_rank == 0)
    {
        savebmp("after.bmp", raw_image, XSIZE, YSIZE);
        free(raw_image);
    }

    // Free up the memory and finalize the MPI enviroment
    free(image);
    MPI_Finalize();
    return 0;
}

// Alteration function creating chessboard layout
void alter(uchar *image, int rank)
{
    int count = rank * block_size;
    for (int y = 0; y < YSIZE; y++)
    {
        for (int x = 0; x < XSIZE * 3; x += 3)
        {
            int alteration = count % 200 < 100 ? 20 : -20;
            if (x % 600 > 300)
                alteration = -alteration;
            *(image + (y * XSIZE * 3) + x) += alteration;
            *(image + (y * XSIZE * 3) + (x + 1)) += alteration;
            *(image + (y * XSIZE * 3) + (x + 2)) += alteration;
        }
        count++;
    }
}