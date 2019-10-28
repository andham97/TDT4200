#include "libs/bitmap.h"
#include <getopt.h>
#include <mpi.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define uchar unsigned char

// Convolutional Kernel Examples, each with dimension 3,
// gaussian kernel with dimension 5
// If you apply another kernel, remember not only to exchange
// the kernel but also the kernelFactor and the correct dimension.

int const sobelYKernel[] = {-1, -2, -1, 0, 0, 0, 1, 2, 1};
float const sobelYKernelFactor = (float)1.0;

int const sobelXKernel[] = {-1, -0, -1, -2, 0, -2, -1, 0, -1, 0};
float const sobelXKernelFactor = (float)1.0;

int const laplacian1Kernel[] = {-1, -4, -1, -4, 20, -4, -1, -4, -1};

float const laplacian1KernelFactor = (float)1.0;

int const laplacian2Kernel[] = {0, 1, 0, 1, -4, 1, 0, 1, 0};
float const laplacian2KernelFactor = (float)1.0;

int const laplacian3Kernel[] = {-1, -1, -1, -1, 8, -1, -1, -1, -1};
float const laplacian3KernelFactor = (float)1.0;

// Bonus Kernel:

int const gaussianKernel[] = {1,  4, 6, 4,  1,  4,  16, 24, 16, 4, 6, 24, 36,
                              24, 6, 4, 16, 24, 16, 4,  1,  4,  6, 4, 1};

float const gaussianKernelFactor = (float)1.0 / 256.0;

// Helper function to swap bmpImageChannel pointers

void swapImageChannel(bmpImageChannel **one, bmpImageChannel **two) {
    bmpImageChannel *helper = *two;
    *two = *one;
    *one = helper;
}

// Apply convolutional kernel on image data
void applyKernel(unsigned char **out, unsigned char **in, unsigned int width,
                 unsigned int height, int *kernel, unsigned int kernelDim,
                 float kernelFactor) {
    unsigned int const kernelCenter = (kernelDim / 2);
    for (unsigned int y = 0; y < height; y++) {
        for (unsigned int x = 0; x < width; x++) {
            int aggregate = 0;
            for (unsigned int ky = 0; ky < kernelDim; ky++) {
                int nky = kernelDim - 1 - ky;
                for (unsigned int kx = 0; kx < kernelDim; kx++) {
                    int nkx = kernelDim - 1 - kx;

                    int yy = y + (ky - kernelCenter);
                    int xx = x + (kx - kernelCenter);
                    if (xx >= 0 && xx < (int)width && yy >= 0 &&
                        yy < (int)height)
                        aggregate += in[yy][xx] * kernel[nky * kernelDim + nkx];
                }
            }
            aggregate *= kernelFactor;
            if (aggregate > 0) {
                out[y][x] = (aggregate > 255) ? 255 : aggregate;
            } else {
                out[y][x] = 0;
            }
        }
    }
}

void help(char const *exec, char const opt, char const *optarg) {
    FILE *out = stdout;
    if (opt != 0) {
        out = stderr;
        if (optarg) {
            fprintf(out, "Invalid parameter - %c %s\n", opt, optarg);
        } else {
            fprintf(out, "Invalid parameter - %c\n", opt);
        }
    }
    fprintf(out, "%s [options] <input-bmp> <output-bmp>\n", exec);
    fprintf(out, "\n");
    fprintf(out, "Options:\n");
    fprintf(out,
            "  -i, --iterations <iterations>    number of iterations (1)\n");

    fprintf(out, "\n");
    fprintf(out, "Example: %s in.bmp out.bmp -i 10000\n", exec);
}

/**
 * Flatten the two-dimensional array from the image channel data to a one
 * dimensional row first array
 */
void flatten(bmpImageChannel *channel, uchar *buffer) {
    for (int y = 0; y < channel->height; y++) {
        for (int x = 0; x < channel->width; x++) {
            buffer[(y * channel->width) + x] = channel->data[y][x];
        }
    }
}

/**
 * Reverse the flatten process taking the one-dimensional row first array
 * "buffer" and storing the values in the two-dimensional array in
 * channel->data.
 *
 * Given that the image channel is correctlyh set up with width and height
 */
void roughen(uchar *buffer, bmpImageChannel *channel) {
    for (int y = 0; y < channel->height; y++) {
        for (int x = 0; x < channel->width; x++) {
            channel->data[y][x] = buffer[(y * channel->width) + x];
        }
    }
}

/**
 * Insert the "from" buffer into the "to" buffer at the offset position, needs
 * "from"s length + offset <= "to"s length
 */
void insert_buffer_at(uchar *from, uchar *to, int offset, int length) {
    for (int i = 0; i < length; i++) {
        to[i + offset] = from[i];
    }
}

/**
 * Extract the data from the "from" buffer at offset and length number of
 * elements, requires "to"s length >= length and "from"s length >= offset +
 * length
 */
void extract_buffer_from(uchar *from, uchar *to, int offset, int length) {
    for (int i = 0; i < length; i++) {
        to[i] = from[i + offset];
    }
}

/**
 * Border exchange to the south
 * The rank = 0, only sends as it will not receive from above, the other recv
 * from above, and the last one will not send to below
 */
void exchange_south(int world_rank, int world_size, bmpImageChannel *channel) {
    if (world_rank != 0) {
        MPI_Recv(channel->data[0], channel->width, MPI_BYTE, world_rank - 1, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    if (world_rank + 1 < world_size) {
        MPI_Send(channel->data[channel->height - 2], channel->width, MPI_BYTE,
                 world_rank + 1, 0, MPI_COMM_WORLD);
    }
}

/**
 * The opposite of the south exchange, sends chained upwards through the ranks,
 * rank = 0 only recv and the rank = world_size - 1 only sends
 */
void exchange_north(int world_rank, int world_size, bmpImageChannel *channel) {
    if (world_rank + 1 < world_size) {
        MPI_Recv(channel->data[channel->height - 1], channel->width, MPI_BYTE,
                 world_rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    if (world_rank != 0) {
        MPI_Send(channel->data[1], channel->width, MPI_BYTE, world_rank - 1, 0,
                 MPI_COMM_WORLD);
    }
}

int main(int argc, char **argv) {
    // Get the clock cycles since the program launched for timing
    clock_t start = clock();

    // Initialize MPI enviroment and fetching size and rank
    MPI_Init(&argc, &argv);
    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    /*
      Parameter parsing, don't change this!
     */
    unsigned int iterations = 1;
    char *output = NULL;
    char *input = NULL;
    int ret = 0;
    static struct option const long_options[] = {
        {"help", no_argument, 0, 'h'},
        {"iterations", required_argument, 0, 'i'},
        {0, 0, 0, 0}};

    static char const *short_options = "hi:";
    {
        char *endptr;
        int c;
        int option_index = 0;
        while ((c = getopt_long(argc, argv, short_options, long_options,
                                &option_index)) != -1) {
            switch (c) {
            case 'h':
                help(argv[0], 0, NULL);
                goto graceful_exit;
            case 'i':
                iterations = strtol(optarg, &endptr, 10);
                if (endptr == optarg) {
                    help(argv[0], c, optarg);
                    goto error_exit;
                }
                break;
            default:
                abort();
            }
        }
    }

    if (argc <= (optind + 1)) {
        help(argv[0], ' ', "Not enough arugments");
        goto error_exit;
    }
    input = calloc(strlen(argv[optind]) + 1, sizeof(char));
    strncpy(input, argv[optind], strlen(argv[optind]));
    optind++;

    output = calloc(strlen(argv[optind]) + 1, sizeof(char));
    strncpy(output, argv[optind], strlen(argv[optind]));
    optind++;

    /*
      End of Parameter parsing!
     */

    bmpImage *image = newBmpImage(0, 0);
    bmpImageChannel *imageChannel;
    int *imageSize = malloc(2 * sizeof(int));
    /*
      Create the BMP image and load it from disk.
      Only the root process load the image for distribution
     */
    if (world_rank == 0) {
        if (image == NULL) {
            fprintf(stderr, "Could not allocate new image!\n");
        }

        if (loadBmpImage(image, input) != 0) {
            fprintf(stderr, "Could not load bmp image '%s'!\n", input);
            freeBmpImage(image);
            goto error_exit;
        }

        imageSize[0] = image->width;
        imageSize[1] = image->height;

        // Create a single color channel image. It is easier to work just with
        // one color
        imageChannel = newBmpImageChannel(image->width, image->height);
        if (imageChannel == NULL) {
            fprintf(stderr, "Could not allocate new image channel!\n");
            freeBmpImage(image);
            goto error_exit;
        }

        // Extract from the loaded image an average over all colors - nothing
        // else than a black and white representation extractImageChannel and
        // mapImageChannel need the images to be in the exact same dimensions!
        // Other prepared extraction functions are extractRed, extractGreen,
        // extractBlue
        if (extractImageChannel(imageChannel, image, extractAverage) != 0) {
            fprintf(stderr, "Could not extract image channel!\n");
            freeBmpImage(image);
            freeBmpImageChannel(imageChannel);
            goto error_exit;
        }
    }

    /**
     * Broadcast the image size to all ranks in order for row calculations
     * when working with row splitted image
     */
    MPI_Bcast(imageSize, 2, MPI_INT, 0, MPI_COMM_WORLD);

    /**
     * Calculate the number of elements to scatter for each rank
     */
    int *counts = malloc(world_size * sizeof(int));
    {
        int sum = 0;
        for (int i = 0; i < world_size - 1; i++) {
            counts[i] = (imageSize[1] / world_size) * imageSize[0];
            sum += imageSize[1] / world_size;
        }
        counts[world_size - 1] = (imageSize[1] - sum) * imageSize[0];
    }

    /**
     * Calculate the number of bytes each rank is displaced in the buffer to
     * scatter
     */
    int *displacements = malloc(world_size * sizeof(int));

    {
        int sum = 0;
        for (int i = 0; i < world_size; i++) {
            displacements[i] = sum;
            sum += counts[i];
        }
    }

    /**
     * Flatten the image channel and send the entire image scattered over the
     * different ranks based on the counts and displacements.
     */
    uchar *send_buffer = NULL;
    uchar *recv_buffer = malloc(counts[world_rank]);
    if (world_rank == 0) {
        send_buffer = malloc(imageSize[0] * imageSize[1]);
        flatten(imageChannel, send_buffer);
    }

    MPI_Scatterv(send_buffer, counts, displacements, MPI_BYTE, recv_buffer,
                 counts[world_rank], MPI_BYTE, 0, MPI_COMM_WORLD);

    // Allocate space for individual parts of the image with space for one halo
    // line north and south
    uchar *process_buffer = malloc(counts[world_rank] + (imageSize[0] * 2));

    // Insert the received buffer into the process buffer with one line offset
    insert_buffer_at(recv_buffer, process_buffer, imageSize[0],
                     counts[world_rank]);

    bmpImageChannel *specific_channel =
        newBmpImageChannel(imageSize[0], counts[world_rank] / imageSize[0] + 2);

    // Remake the 2-dim array from the process_buffer with space for more data
    roughen(process_buffer, specific_channel);

    // Here we do the actual computation!
    // imageChannel->data is a 2-dimensional array of unsigned char which is
    // accessed row first ([y][x])
    bmpImageChannel *processImageChannel =
        newBmpImageChannel(specific_channel->width, specific_channel->height);
    for (unsigned int i = 0; i < iterations; i++) {
        // Performing the border exchange first south rank 0 through rank n.
        // Then north rank n through rank 0
        exchange_south(world_rank, world_size, specific_channel);
        exchange_north(world_rank, world_size, specific_channel);
        applyKernel(
            processImageChannel->data, specific_channel->data,
            specific_channel->width, specific_channel->height,
            (int *)laplacian1Kernel, 3, laplacian1KernelFactor
            //               (int *)laplacian2Kernel, 3, laplacian2KernelFactor
            //               (int *)laplacian3Kernel, 3, laplacian3KernelFactor
            //               (int *)gaussianKernel, 5, gaussianKernelFactor
        );
        swapImageChannel(&processImageChannel, &specific_channel);
    }
    freeBmpImageChannel(processImageChannel);

    // Make the 2-dim array into a 1-dim array for ease of sending
    flatten(specific_channel, process_buffer);

    // Extract the result buffer from the process_buffer ignoring halo rows
    extract_buffer_from(process_buffer, recv_buffer, imageSize[0],
                        counts[world_rank]);

    // Gathering the buffers back at the root process for final assembly
    MPI_Gatherv(recv_buffer, counts[world_rank], MPI_BYTE, send_buffer, counts,
                displacements, MPI_BYTE, 0, MPI_COMM_WORLD);

    // The data is placed at the root process, so this will do the assembly and
    // storing the result to disk
    if (world_rank == 0) {
        // Insert the received image buffers into the imageChannel
        roughen(send_buffer, imageChannel);
        // Map our single color image back to a normal BMP image with 3 color
        // channels mapEqual puts the color value on all three channels the same
        // way other mapping functions are mapRed, mapGreen, mapBlue
        if (mapImageChannel(image, imageChannel, mapEqual) != 0) {
            fprintf(stderr, "Could not map image channel!\n");
            freeBmpImage(image);
            freeBmpImageChannel(imageChannel);
            goto error_exit;
        }
        freeBmpImageChannel(imageChannel);

        // Write the image back to disk
        if (saveBmpImage(image, output) != 0) {
            fprintf(stderr, "Could not save output to '%s'!\n", output);
            freeBmpImage(image);
            goto error_exit;
        };
    }

    // Finalizing MPI
    MPI_Finalize();

graceful_exit:
    ret = 0;
error_exit:
    if (input)
        free(input);
    if (output)
        free(output);

    // Get the number of clock cycles since launch and calculating the time
    // elapsed for execution
    clock_t end = clock();
    double time_spent = ((double)(end - start) / CLOCKS_PER_SEC) * 1000;
    printf("Rank %d - %fms\n", world_rank, time_spent);
    return ret;
};
