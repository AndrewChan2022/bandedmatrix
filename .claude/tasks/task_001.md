

implement:

    banded matrix solver for LU and QR

    based on chapter 2.4 Tridiagonal and Band-Diagonal Systems of Equations
        the book in books/
        high-perf skills in skills/high-perf

verify:

    also implement dense matrix LU and QR.

    then benchmark dense and banded matrix.

    test data can be:
        a. small matrix.
        b. middle scale matrix.   1000 * 1000,   band matrix has band 3
        d. large scale matrix.  1m * 1m, band matrix has band 3.

report:

    after test and benchmark

    write summary document and report to docs/