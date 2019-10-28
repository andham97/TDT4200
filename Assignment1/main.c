#include <stdlib.h>
#include <stdio.h>
#include "bitmap.h"

#define XSIZE 2560 // Size of before image
#define YSIZE 2048

uchar *resize(uchar *image, int size);
void alter(uchar *image);

int main(void)
{
    int size = 2;
    uchar *image = malloc(XSIZE * YSIZE * 3); // Three uchars per pixel (RGB)
    readbmp("before.bmp", image);

    alter(image);
    image = resize(image, size);

    savebmp("after.bmp", image, XSIZE * size, YSIZE * size);

    free(image);
    return 0;
}

uchar *resize(uchar *image, int size)
{
    // Allocating size for the resized image
    uchar *buffer = malloc(XSIZE * YSIZE * size * size * 3);
    int newXSize = XSIZE * size;
    int newYSize = YSIZE * size;

    // Looping all pixels, and i,j loops for adding the new pixels with copying pixels given by x and y from original, then placing them in the sizeXsize "grid" the new enlarged image use for same pixels
    for (int y = 0; y < YSIZE; y++)
    {
        for (int x = 0; x < XSIZE * 3; x += 3)
        {
            for (int i = 0; i < size; i++)
            {
                for (int j = 0; j < size * 3; j += 3)
                {
                    *(buffer + (y * newXSize * 3 * size) + ((x * size) + (newXSize * 3 * i) + j)) = *(image + (y * XSIZE * 3) + x);
                    *(buffer + (y * newXSize * 3 * size) + ((x * size) + (newXSize * 3 * i) + j + 1)) = *(image + (y * XSIZE * 3) + x + 1);
                    *(buffer + (y * newXSize * 3 * size) + ((x * size) + (newXSize * 3 * i) + j + 2)) = *(image + (y * XSIZE * 3) + x + 2);
                }
            }
        }
    }
    return buffer;
}

void alter(uchar *image)
{
    uchar temp[3];
    // Altering image by removing green from all pixels
    for (int y = 0; y < YSIZE; y++)
    {
        for (int x = 0; x < XSIZE * 3; x += 3)
        {
            *(image + (y * XSIZE * 3) + (x + 1)) = 0;
        }
    }
}