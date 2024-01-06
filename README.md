# Giga-Flops
A test in different languages to get the highest amount of computational power outof the programs.

### But, What is it?
Matrix Multiplication, Perform Matrix multiplication and try to beat the benchmark set by the famous Python Library **Numpy**. 
All the programs implement different method to beat the Numpy's performance.
\**More Details at the end*

### How to compile

#### C
There are different types algorithms implemented.
* Naive
* Cache
* SIMD (Fast)
* Parallel Processing (PP)

```
cc -march=native -D[TYPE] -pthread -o x.out c.c && ./x.out
```
eg. `cc -march=native -DCACHE -pthread -o x.out c.c && ./x.out`
for **CACHE** running the cache version.


#### Python 
There are many different types algorithms implemented.
* Numpy
```
python3 python.py
```

#### D-Lang
*Work Under progress*


### More Details
The `mat.dat` files contain data of 2 matrices, `Matrix VAL` and `Matrix RES`
Both of them are `1024x1024`. The relation between them is `VAL*VAL = RES`
where (\*) is **Matrix Multiplication**.

These matrices are stored in a very simple format, encoding all `1024*1024`
floats of `Matrix VAL` into bytes and then writing them into the file, followed by`Matrix RES`

These matrices are given so that the result can be verified.
