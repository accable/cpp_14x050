#include <iostream>

// Creating a 13x13 matrix of 1's
const int matrix_s = 13;


// Just to print the matrix
void printMatrix(double matrix[matrix_s][matrix_s]){
    for(int i = 0; i < matrix_s; i++){
        for(int j = 0; j < matrix_s; j++){
            std::cout << matrix[i][j];
        }
        std::cout << std::endl;
    }
}


// Where the magic is 
int main(){

    double matrix[matrix_s][matrix_s];
    int col_idx = 1;
    int row_idx = 1;
    int mid_col_idx = 3;
    int mid_row_idx = 3;

    // Instead of initializing the matrix, we simply created it on the spot
    // Why? Because we can.
    for(int i = 0; i < matrix_s; i++){

        // We use index tracking to make sure that the transformation is done on correct rows
        // This means we use "determined indexes" to make sure the transformation hits the correct 
        // row and columns.
        if(i != row_idx){  // If the row is not on the idx
            for (int j = 0; j < matrix_s; j++){

                // For all column, if j is 1 or 6, add by 5
                if(j == col_idx){
                    matrix[i][j] = 2;
                    col_idx+= 5; 
                } else {
                    matrix[i][j] = 1;                    
                }
            }        
        
        col_idx = 1;  // On row change, we reset the index
        
        // For all rows, if i is 1 or 6, add by 5
        } else if (i == row_idx){
            for(int j = 0; j < matrix_s; j++){
                matrix[i][j] = 2;
            }
        row_idx+= 5;  // Instead of resetting the index, we add it instead
        }
    }
    
    // Now we add the middle 3's since the "general shape" has been created
    for(int i = 0; i < matrix_s; i++){

        // We only do it on mid index, which is 3 and 4
        if(i == mid_row_idx){
            for(int j = 0; j < matrix_s; j++){
                if(j == mid_col_idx){
                    matrix[i][j] = 3;
                    mid_col_idx+= 1;  // So we cover both 3 and 4
                }

                if(mid_col_idx == 5){
                    mid_col_idx+= 3;
                } else if(mid_col_idx == 10){
                    mid_col_idx = 3;
                }
                    
            }
        
            mid_row_idx+= 1;

            if(mid_row_idx == 5){
                mid_row_idx+= 3;
            } else if (mid_row_idx == 10){
                break;
            }
        }
    }
    printMatrix(matrix);
    return 0;
}