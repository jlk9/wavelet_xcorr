
#include <stdlib.h>
#include<stdint.h>

/* Once we get indices we need to increment (provided in indices_left and indices_right), we add them to our xcorr values.
*/
double* sparse_xcorr_sum(const int64_t* indices_left, const int64_t* indices_right, const int64_t* sparse_indices1, const int64_t* sparse_indices2,
                         const double* sparse_values1, const double* sparse_values2, const int lagmax, const int length){

    double* sparse_xcorrs = (double*) calloc(lagmax, sizeof(double));

    // Sums up the sparse terms into the actual xcorrs:
    for (int i=0; i<length; ++i){

        int64_t sparse_index2 = sparse_indices2[i];
        double sparse_value2  = sparse_values2[i];

        for (int j=indices_left[i]; j<indices_right[i]; ++j){

            sparse_xcorrs[sparse_indices1[j] - sparse_index2] += sparse_values1[j] * sparse_value2;
        }
    }
    
    return sparse_xcorrs;
}


void sparse_xcorr_sum_void(const int64_t* indices_left, const int64_t* indices_right, const int64_t* sparse_indices1, const int64_t* sparse_indices2,
                         const double* sparse_values1, const double* sparse_values2, const int length, double* sparse_xcorrs){

    // Sums up the sparse terms into the actual xcorrs:
    for (int i=0; i<length; ++i){

        int64_t sparse_index2 = sparse_indices2[i];
        double sparse_value2  = sparse_values2[i];

        for (int j=indices_left[i]; j<indices_right[i]; ++j){

            sparse_xcorrs[sparse_indices1[j] - sparse_index2] += sparse_values1[j] * sparse_value2;
        }
    }
}


/* Here, we add/subtract a set of entries to a vector at predetermined indices.
    This C function will replace the inner loop in the subdiagonal computations, allowing us to quickly decrement
    diagonal values by the necessary amounts at the desired indices.

    For now, it looks like we don't need this.
*/
//void sparse_vector_sum(int64_t* target_array){

    // add here

//}