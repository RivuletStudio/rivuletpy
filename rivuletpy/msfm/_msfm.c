#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include "_msfm.h"

#define eps 2.2204460492503131e-16
#define doublemax 1e50
#define INF 2e50
#define listINF 2.345e50
#ifndef min
#define min(a,b)        ((a) < (b) ? (a): (b))
#endif
#ifndef max
#define max(a,b)        ((a) > (b) ? (a): (b))
#endif

/* Find minimum value of an array and return its index */
int minarray(double *A, int l) {
    int i;
    int minind=0;
    for (i=0; i<l; i++) { if(A[i]<A[minind]){ minind=i; } }
    return minind;
}

double pow2(double val) { return val*val; }

int iszero(double a){ return a*a<eps; }
int isnotzero(double a){ return a*a>eps; }

void roots(double* Coeff, double* ans) {
    double a=Coeff[0];
    double b=Coeff[1];
    double c=Coeff[2];
    double r1, r2;
    double d;
    
    d=max(pow2(b)-4.0*a*c,0);
    if(isnotzero(a)) {
        ans[0]= (-b - sqrt(d)) / (2.0*a);
        ans[1]= (-b + sqrt(d)) / (2.0*a);
    }
    else {
    r1=(-b - sqrt(d));
    r2=(-b + sqrt(d));
    if(isnotzero(r1))
    {
      if(isnotzero(r2))
      {
        ans[0]= (2.0*c)/r1; ans[1]= (2.0*c)/r2;
      }
      else
      {
        ans[0]= (2.0*c)/r1; ans[1]= (2.0*c)/r1;
      }
    }
    else if(isnotzero(r2))
    {
      ans[0]= (2.0*c)/r2; ans[1]= (2.0*c)/r2;
    }
    else
    {
      ans[0]=0; ans[1]=0;
    }
    }
}


int maxarray(double *A, int l) {
    int i;
    int maxind=0;
    for (i=0; i<l; i++) { if(A[i]>A[maxind]){ maxind=i; } }
    return maxind;
}

__inline int mindex3(int x, int y, int z, int sizx, int sizy) { return x+y*sizx+z*sizx*sizy; }


__inline bool IsFinite(double x) { return (x <= doublemax  && x >= -doublemax ); }
__inline bool IsInf(double x)    { return (x >= doublemax ); }

__inline bool IsListInf(double x){ return (x == listINF ); }


__inline bool isntfrozen3d(int i, int j, int k, int *dims, bool *Frozen) {

    return (i>=0)&&(j>=0)&&(k>=0)&&(i<dims[0])&&(j<dims[1])&&(k<dims[2])&&(Frozen[mindex3(i, j, k, dims[0], dims[1])]==0);
}
__inline bool isfrozen3d(int i, int j, int k, int *dims, bool *Frozen) {

    return (i>=0)&&(j>=0)&&(k>=0)&&(i<dims[0])&&(j<dims[1])&&(k<dims[2])&&(Frozen[mindex3(i, j, k, dims[0], dims[1])]==1);
}

int p2x(int x) /* 2^x */
{
/*    return pow(2,x); */
    int y=1;
    int p2x[16] ={1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768};
    while(x>15) { x=x-15; y=y*32768; }
    return y*p2x[x];
}

void show_list(double **listval, int *listprop) {
    int z, k;
    for(z=0;z<listprop[1]; z++) {
        for(k=0;k<p2x(z+1); k++) {
            if((z>0)&&(listval[z-1][(int)floor(k/2)]>listval[z][k])) {
                printf("*%15.5f", listval[z][k]);
            }
            else {
                printf(" %15.5f", listval[z][k]);
            }
        }
        printf("\n");
    }
}

void initialize_list(double ** listval, int *listprop) {
    /* Loop variables */
    int i;
    /* Current Length, Orde and Current Max Length */
    listprop[0]=0; listprop[1]=1; listprop[2]=2;
    /* Make first orde storage of 2 values */
    listval[0]=(double*)malloc(2 * sizeof(double));
    /* Initialize on infinite */
    for(i=0;i<2;i++) { listval[0][i]=listINF; }
}


void destroy_list(double ** listval, int *listprop) {
    /* Loop variables */
    int i, list_orde;
    /* Get list orde */
    list_orde=listprop[1];
    /* Free memory */
    for(i=0;i<list_orde;i++) { free(listval[i]); }
    free(listval);
    free(listprop);
}

void list_add(double ** listval, int *listprop, double val) {
    /* List parameters */
    int list_length, list_orde, list_lengthmax;
    /* Loop variable */
    int i, j;
    /* Temporary list location */
    int listp;
    /* Get list parameters */
    list_length=listprop[0]; list_orde=listprop[1]; list_lengthmax=listprop[2];
    /* If list is full expand list */
    if(list_length==list_lengthmax) {
        list_lengthmax=list_lengthmax*2;
        for (i=list_orde; i>0; i--) {
            listval[i]=listval[i-1];
            listval[i] = (double *)realloc(listval[i], p2x(i+1)*sizeof(double));
            for(j=p2x(i); j<p2x(i+1); j++) { listval[i][j]=listINF;  }
        }
        listval[0]=(double *)malloc(2*sizeof(double));
        listval[0][0]=min(listval[1][0], listval[1][1]);
        listval[0][1]=listINF;
        list_orde++;
    }
    /* Add a value to the list */
    listp=list_length;
    list_length++;
    listval[list_orde-1][listp]=val;
    /* Update the links minimum */
    for (i=list_orde-1; i>0; i--) {
        listp=(int)floor(((double)listp)/2);
        if(val<listval[i-1][listp]) { listval[i-1][listp]=val; } else { break; }
    }
    /* Set list parameters */
    listprop[0]=list_length; listprop[1]=list_orde; listprop[2]=list_lengthmax;
}

int list_minimum(double ** listval, int *listprop) {
    /* List parameters */
    int list_orde = listprop[1];
    /* Loop variable */
    int i;
    /* Temporary list location */
    int listp;
    /* Index of Minimum */
    int minindex;

    /* Follow the minimum through the binary tree */
    listp=0;
    for(i=0;i<(list_orde-1);i++) {
        
        /* <= ????? */ 
        if(listval[i][listp]<=listval[i][listp+1]) { listp=listp*2; } else { listp=(listp+1)*2; }
    }
    i=list_orde-1;
    if(listval[i][listp]<=listval[i][listp+1]){minindex=listp; } else { minindex=listp+1; }
    return minindex;
}


void list_remove(double ** listval, int *listprop, int index) {
    /* List parameters */
    int list_orde;
    /* Loop variable */
    int i;
    /* Temp index */
    int index2;
    double val;
    double valmin;
    /* Get list parameters */
    list_orde=listprop[1];
    /* Temporary store current value */
    val=listval[list_orde-1][index];
    valmin=listINF;
    /* Replace value by infinite */
    listval[list_orde-1][index]=listINF;
    /* Follow the binary tree to replace value by minimum values from */
    /** the other values in the binary tree. */

    i=list_orde-1;
    while(true) {
        if((index%2)==0) { index2=index+1; } else { index2=index-1; }
        if(val<listval[i][index2]) {
            index=(int)floor(((double)index2)/2.0);
            if(listval[i][index2]<valmin) { valmin=listval[i][index2]; }
      if(i==0) { break; }
            listval[i-1][index]=valmin;
            i--; if(i==0) { break; }
        }
        else { break; }
    }
}

void list_remove_replace(double ** listval, int *listprop, int index) {
    /* List parameters */
    int list_length, list_orde, list_lengthmax;
    /* Loop variable */
    int i, listp;
    /* Temporary store value */
    double val;
    int templ;
    /* Get list parameters */
    list_length=listprop[0];
    list_orde=listprop[1];
    list_lengthmax=listprop[2];
    /* Remove the value */
    list_remove(listval, listprop, index);
    /* Replace the removed value by the last in the list. (if it was */
    /* not already the last value) */
    if(index<(list_length-1)) {
        /* Temporary store last value in the list */
        val=listval[list_orde-1][list_length-1];
        /* Remove last value in the list */
        list_remove(listval, listprop, list_length-1);
        /* Add a value to the list */
        listp=index;
        listval[list_orde-1][index]=val;
        /* Update the links minimum */
        for (i=(list_orde-1); i>0; i--) {
            listp=(int)floor(((double)listp)/2);
            if(val<listval[i-1][listp]) { listval[i-1][listp]=val; } else {  break; }
        }
    }
    /* List is now shorter */
    list_length--;
    /* Remove trunk of binary tree  / Free memory if list becomes shorter */
    if(list_orde>2&&IsListInf(listval[0][1])) {
        list_orde--;
        list_lengthmax=list_lengthmax/2;
        /* Remove trunk array */
        free(listval[0]);
        /* Move the other arrays one up */
        templ=2;
        for (i=0; i<list_orde; i++) {
            listval[i]=listval[i+1];
            /* Resize arrays to their new shorter size */
            listval[i] = (double *)realloc(listval[i], templ*sizeof(double));
            templ*=2;
        }
    }
    /* Set list parameters */
    listprop[0]=list_length; listprop[1]=list_orde; listprop[2]=list_lengthmax;
}

void listupdate(double **listval, int *listprop, int index, double val) {
    /* List parameters */
    int list_length, list_orde, list_lengthmax;
    /* loop variable */
    int i, listp;
    /* Get list parameters */
    list_length=listprop[0];
    list_orde=listprop[1];
    list_lengthmax=listprop[2];
    /* Add a value to the list */
    listp=index;
    listval[list_orde-1][index]=val;
    /* Update the links minimum */
    for (i=(list_orde-1); i>0; i--) {
        listp=(int)floor(((double)listp)/2);
        if(val<listval[i-1][listp]) { listval[i-1][listp]=val; } else { break; }
    }
    /* Set list parameters */
    listprop[0]=list_length; listprop[1]=list_orde; listprop[2]=list_lengthmax;
}

__inline int mindex2(int x, int y, int sizx) { return x+y*sizx; }    

__inline bool isntfrozen2d(int i, int j, int *dims, bool *Frozen)
{
    return (i>=0)&&(j>=0)&&(i<dims[0])&&(j<dims[1])&&(Frozen[i+j*dims[0]]==0);
}
__inline bool isfrozen2d(int i, int j, int *dims, bool *Frozen)
{
    return (i>=0)&&(j>=0)&&(i<dims[0])&&(j<dims[1])&&(Frozen[i+j*dims[0]]==1);
}

npy_double second_derivative(npy_double Txm1, npy_double Txm2, npy_double Txp1, npy_double Txp2) {
  int ch1, ch2;
  npy_double Tm;
  Tm = INF;
  ch1 = (Txm2 < Txm1) && IsFinite(Txm1);
  ch2 = (Txp2 < Txp1) && IsFinite(Txp1);
  if (ch1 && ch2) {
    Tm = min((4.0 * Txm1 - Txm2) / 3.0, (4.0 * Txp1 - Txp2) / 3.0);
  } else if (ch1) {
    Tm = (4.0 * Txm1 - Txm2) / 3.0;
  } else if (ch2) {
    Tm = (4.0 * Txp1 - Txp2) / 3.0;
  }
  return Tm;
}

npy_double CalculateDistance(npy_double* T, npy_double Fijk, int* dims, int i, int j, int k,
                         int usesecond, int usecross, bool* Frozen) {
  /* Loop variables */
  int q, t;

  /* Current location */
  int in, jn, kn;

  /* Derivatives */
  npy_double Tm[18] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  npy_double Tm2[18] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  npy_double Coeff[3];

  /* local derivatives in distance image */
  npy_double Txm1, Txm2, Txp1, Txp2 = 0;
  npy_double Tym1, Tym2, Typ1, Typ2 = 0;
  npy_double Tzm1, Tzm2, Tzp1, Tzp2 = 0;
  /* local cross derivatives in distance image */
  // npy_double Tr2t1m1, Tr2t1m2, Tr2t1p1, Tr2t1p2;
  npy_double Tr2t2m1, Tr2t2m2, Tr2t2p1, Tr2t2p2 = 0.0;
  npy_double Tr2t3m1, Tr2t3m2, Tr2t3p1, Tr2t3p2 = 0.0;
  // npy_double Tr3t1m1, Tr3t1m2, Tr3t1p1, Tr3t1p2;
  npy_double Tr3t2m1, Tr3t2m2, Tr3t2p1, Tr3t2p2 = 0.0;
  npy_double Tr3t3m1, Tr3t3m2, Tr3t3p1, Tr3t3p2 = 0.0;
  // npy_double Tr4t1m1, Tr4t1m2, Tr4t1p1, Tr4t1p2;
  npy_double Tr4t2m1, Tr4t2m2, Tr4t2p1, Tr4t2p2 = 0.0;
  npy_double Tr4t3m1, Tr4t3m2, Tr4t3p1, Tr4t3p2 = 0.0;
  // npy_double Tr5t1m1, Tr5t1m2, Tr5t1p1, Tr5t1p2;
  npy_double Tr5t2m1, Tr5t2m2, Tr5t2p1, Tr5t2p2 = 0.0;
  npy_double Tr5t3m1, Tr5t3m2, Tr5t3p1, Tr5t3p2 = 0.0;
  // npy_double Tr6t1m1, Tr6t1m2, Tr6t1p1
  // npy_double Tr6t1p2;
  npy_double Tr6t2m1, Tr6t2m2, Tr6t2p1, Tr6t2p2 = 0.0;
  npy_double Tr6t3m1, Tr6t3m2, Tr6t3p1, Tr6t3p2 = 0.0;

  npy_double Tt, Tt2 = 0.0;

  /* Return values root of polynomial */
  npy_double ansroot[2] = {0, 0};

  /* Order derivatives in a certain direction */
  int Order[18] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

  /* Neighbours 4x2 */
  // int ne[18] = {-1, 0, 0, 1, 0, 0, 0, -1, 0, 0, 1, 0, 0, 0, -1, 0, 0, 1};

  /* Stencil constants */
  npy_double G1[18] = {1,
                   1,
                   1,
                   1,
                   0.5,
                   0.5,
                   1,
                   0.5,
                   0.5,
                   1,
                   0.5,
                   0.5,
                   0.5,
                   0.3333333333333,
                   0.3333333333333,
                   0.5,
                   0.3333333333333,
                   0.3333333333333};
  npy_double G2[18] = {2.250, 2.250, 2.250, 2.250, 1.125, 1.125,
                   2.250, 1.125, 1.125, 2.250, 1.125, 1.125,
                   1.125, 0.750, 0.750, 1.125, 0.750, 0.750};

  /*Get First order derivatives (only use frozen pixel) */
  in = i - 1;
  jn = j + 0;
  kn = k + 0;
  if (isfrozen3d(in, jn, kn, dims, Frozen)) {
    Txm1 = T[mindex3(in, jn, kn, dims[0], dims[1])];
  } else {
    Txm1 = INF;
  }
  in = i + 1;
  jn = j + 0;
  kn = k + 0;
  if (isfrozen3d(in, jn, kn, dims, Frozen)) {
    Txp1 = T[mindex3(in, jn, kn, dims[0], dims[1])];
  } else {
    Txp1 = INF;
  }
  in = i + 0;
  jn = j - 1;
  kn = k + 0;
  if (isfrozen3d(in, jn, kn, dims, Frozen)) {
    Tym1 = T[mindex3(in, jn, kn, dims[0], dims[1])];
  } else {
    Tym1 = INF;
  }
  in = i + 0;
  jn = j + 1;
  kn = k + 0;
  if (isfrozen3d(in, jn, kn, dims, Frozen)) {
    Typ1 = T[mindex3(in, jn, kn, dims[0], dims[1])];
  } else {
    Typ1 = INF;
  }
  in = i + 0;
  jn = j + 0;
  kn = k - 1;
  if (isfrozen3d(in, jn, kn, dims, Frozen)) {
    Tzm1 = T[mindex3(in, jn, kn, dims[0], dims[1])];
  } else {
    Tzm1 = INF;
  }
  in = i + 0;
  jn = j + 0;
  kn = k + 1;
  if (isfrozen3d(in, jn, kn, dims, Frozen)) {
    Tzp1 = T[mindex3(in, jn, kn, dims[0], dims[1])];
  } else {
    Tzp1 = INF;
  }

  if (usecross) {
    // Tr2t1m1 = Txm1;
    // Tr2t1p1 = Txp1;
    in = i - 0;
    jn = j - 1;
    kn = k - 1;
    if (isfrozen3d(in, jn, kn, dims, Frozen)) {
      Tr2t2m1 = T[mindex3(in, jn, kn, dims[0], dims[1])];
    } else {
      Tr2t2m1 = INF;
    }
    in = i + 0;
    jn = j + 1;
    kn = k + 1;
    if (isfrozen3d(in, jn, kn, dims, Frozen)) {
      Tr2t2p1 = T[mindex3(in, jn, kn, dims[0], dims[1])];
    } else {
      Tr2t2p1 = INF;
    }
    in = i - 0;
    jn = j - 1;
    kn = k + 1;
    if (isfrozen3d(in, jn, kn, dims, Frozen)) {
      Tr2t3m1 = T[mindex3(in, jn, kn, dims[0], dims[1])];
    } else {
      Tr2t3m1 = INF;
    }
    in = i + 0;
    jn = j + 1;
    kn = k - 1;

    if (isfrozen3d(in, jn, kn, dims, Frozen)) {
      Tr2t3p1 = T[mindex3(in, jn, kn, dims[0], dims[1])];
    } else {
      Tr2t3p1 = INF;
    }
    // Tr3t1m1 = Tym1;
    // Tr3t1p1 = Typ1;
    in = i - 1;
    jn = j + 0;
    kn = k + 1;
    if (isfrozen3d(in, jn, kn, dims, Frozen)) {
      Tr3t2m1 = T[mindex3(in, jn, kn, dims[0], dims[1])];
    } else {
      Tr3t2m1 = INF;
    }
    in = i + 1;
    jn = j + 0;
    kn = k - 1;

    if (isfrozen3d(in, jn, kn, dims, Frozen)) {
      Tr3t2p1 = T[mindex3(in, jn, kn, dims[0], dims[1])];
    } else {
      Tr3t2p1 = INF;
    }

    in = i - 1;
    jn = j - 0;
    kn = k - 1;

    if (isfrozen3d(in, jn, kn, dims, Frozen)) {
      Tr3t3m1 = T[mindex3(in, jn, kn, dims[0], dims[1])];
    } else {
      Tr3t3m1 = INF;
    }

    in = i + 1;
    jn = j + 0;
    kn = k + 1;

    if (isfrozen3d(in, jn, kn, dims, Frozen)) {
      Tr3t3p1 = T[mindex3(in, jn, kn, dims[0], dims[1])];
    } else {
      Tr3t3p1 = INF;
    }

    // Tr4t1m1 = Tzm1;
    // Tr4t1p1 = Tzp1;
    in = i - 1;
    jn = j - 1;
    kn = k - 0;
    if (isfrozen3d(in, jn, kn, dims, Frozen)) {
      Tr4t2m1 = T[mindex3(in, jn, kn, dims[0], dims[1])];
    } else {
      Tr4t2m1 = INF;
    }
    in = i + 1;
    jn = j + 1;
    kn = k + 0;
    if (isfrozen3d(in, jn, kn, dims, Frozen)) {
      Tr4t2p1 = T[mindex3(in, jn, kn, dims[0], dims[1])];
    } else {
      Tr4t2p1 = INF;
    }
    in = i - 1;
    jn = j + 1;
    kn = k - 0;
    if (isfrozen3d(in, jn, kn, dims, Frozen)) {
      Tr4t3m1 = T[mindex3(in, jn, kn, dims[0], dims[1])];
    } else {
      Tr4t3m1 = INF;
    }
    in = i + 1;
    jn = j - 1;
    kn = k + 0;
    if (isfrozen3d(in, jn, kn, dims, Frozen)) {
      Tr4t3p1 = T[mindex3(in, jn, kn, dims[0], dims[1])];
    } else {
      Tr4t3p1 = INF;
    }
    // Tr5t1m1 = Tr3t3m1;
    // Tr5t1p1 = Tr3t3p1;
    in = i - 1;
    jn = j - 1;
    kn = k + 1;
    if (isfrozen3d(in, jn, kn, dims, Frozen)) {
      Tr5t2m1 = T[mindex3(in, jn, kn, dims[0], dims[1])];
    } else {
      Tr5t2m1 = INF;
    }
    in = i + 1;
    jn = j + 1;
    kn = k - 1;
    if (isfrozen3d(in, jn, kn, dims, Frozen)) {
      Tr5t2p1 = T[mindex3(in, jn, kn, dims[0], dims[1])];
    } else {
      Tr5t2p1 = INF;
    }
    in = i - 1;
    jn = j + 1;
    kn = k + 1;
    if (isfrozen3d(in, jn, kn, dims, Frozen)) {
      Tr5t3m1 = T[mindex3(in, jn, kn, dims[0], dims[1])];
    } else {
      Tr5t3m1 = INF;
    }
    in = i + 1;
    jn = j - 1;
    kn = k - 1;
    if (isfrozen3d(in, jn, kn, dims, Frozen)) {
      Tr5t3p1 = T[mindex3(in, jn, kn, dims[0], dims[1])];
    } else {
      Tr5t3p1 = INF;
    }
    // Tr6t1m1 = Tr3t2p1;
    // Tr6t1p1 = Tr3t2m1;
    in = i - 1;
    jn = j - 1;
    kn = k - 1;
    if (isfrozen3d(in, jn, kn, dims, Frozen)) {
      Tr6t2m1 = T[mindex3(in, jn, kn, dims[0], dims[1])];
    } else {
      Tr6t2m1 = INF;
    }
    in = i + 1;
    jn = j + 1;
    kn = k + 1;
    if (isfrozen3d(in, jn, kn, dims, Frozen)) {
      Tr6t2p1 = T[mindex3(in, jn, kn, dims[0], dims[1])];
    } else {
      Tr6t2p1 = INF;
    }
    in = i - 1;
    jn = j + 1;
    kn = k - 1;
    if (isfrozen3d(in, jn, kn, dims, Frozen)) {
      Tr6t3m1 = T[mindex3(in, jn, kn, dims[0], dims[1])];
    } else {
      Tr6t3m1 = INF;
    }
    in = i + 1;
    jn = j - 1;
    kn = k + 1;
    if (isfrozen3d(in, jn, kn, dims, Frozen)) {
      Tr6t3p1 = T[mindex3(in, jn, kn, dims[0], dims[1])];
    } else {
      Tr6t3p1 = INF;
    }
  }

  /*The values in order is 0 if no neighbours in that direction */
  /*1 if 1e order derivatives is used and 2 if second order */
  /*derivatives are used */

  /*Make 1e order derivatives in x and y direction */
  Tm[0] = min(Txm1, Txp1);
  if (IsFinite(Tm[0])) {
    Order[0] = 1;
  } else {
    Order[0] = 0;
  }
  Tm[1] = min(Tym1, Typ1);
  if (IsFinite(Tm[1])) {
    Order[1] = 1;
  } else {
    Order[1] = 0;
  }
  Tm[2] = min(Tzm1, Tzp1);
  if (IsFinite(Tm[2])) {
    Order[2] = 1;
  } else {
    Order[2] = 0;
  }

  /*Make 1e order derivatives in cross directions */
  if (usecross) {
    Tm[3] = Tm[0];
    Order[3] = Order[0];
    Tm[4] = min(Tr2t2m1, Tr2t2p1);
    if (IsFinite(Tm[4])) {
      Order[4] = 1;
    } else {
      Order[4] = 0;
    }
    Tm[5] = min(Tr2t3m1, Tr2t3p1);
    if (IsFinite(Tm[5])) {
      Order[5] = 1;
    } else {
      Order[5] = 0;
    }

    Tm[6] = Tm[1];
    Order[6] = Order[1];
    Tm[7] = min(Tr3t2m1, Tr3t2p1);
    if (IsFinite(Tm[7])) {
      Order[7] = 1;
    } else {
      Order[7] = 0;
    }
    Tm[8] = min(Tr3t3m1, Tr3t3p1);
    if (IsFinite(Tm[8])) {
      Order[8] = 1;
    } else {
      Order[8] = 0;
    }

    Tm[9] = Tm[2];
    Order[9] = Order[2];
    Tm[10] = min(Tr4t2m1, Tr4t2p1);
    if (IsFinite(Tm[10])) {
      Order[10] = 1;
    } else {
      Order[10] = 0;
    }
    Tm[11] = min(Tr4t3m1, Tr4t3p1);
    if (IsFinite(Tm[11])) {
      Order[11] = 1;
    } else {
      Order[11] = 0;
    }

    Tm[12] = Tm[8];
    Order[12] = Order[8];
    Tm[13] = min(Tr5t2m1, Tr5t2p1);
    if (IsFinite(Tm[13])) {
      Order[13] = 1;
    } else {
      Order[13] = 0;
    }
    Tm[14] = min(Tr5t3m1, Tr5t3p1);
    if (IsFinite(Tm[14])) {
      Order[14] = 1;
    } else {
      Order[14] = 0;
    }

    Tm[15] = Tm[7];
    Order[15] = Order[7];
    Tm[16] = min(Tr6t2m1, Tr6t2p1);
    if (IsFinite(Tm[16])) {
      Order[16] = 1;
    } else {
      Order[16] = 0;
    }
    Tm[17] = min(Tr6t3m1, Tr6t3p1);
    if (IsFinite(Tm[17])) {
      Order[17] = 1;
    } else {
      Order[17] = 0;
    }
  }

  /*Make 2e order derivatives */
  if (usesecond) {
    /*Get Second order derivatives (only use frozen pixel) */
    /*Get First order derivatives (only use frozen pixel) */
    in = i - 2;
    jn = j + 0;
    kn = k + 0;
    if (isfrozen3d(in, jn, kn, dims, Frozen)) {
      Txm2 = T[mindex3(in, jn, kn, dims[0], dims[1])];
    } else {
      Txm2 = INF;
    }
    in = i + 2;
    jn = j + 0;
    kn = k + 0;
    if (isfrozen3d(in, jn, kn, dims, Frozen)) {
      Txp2 = T[mindex3(in, jn, kn, dims[0], dims[1])];
    } else {
      Txp2 = INF;
    }
    in = i + 0;
    jn = j - 2;
    kn = k + 0;
    if (isfrozen3d(in, jn, kn, dims, Frozen)) {
      Tym2 = T[mindex3(in, jn, kn, dims[0], dims[1])];
    } else {
      Tym2 = INF;
    }
    in = i + 0;
    jn = j + 2;
    kn = k + 0;
    if (isfrozen3d(in, jn, kn, dims, Frozen)) {
      Typ2 = T[mindex3(in, jn, kn, dims[0], dims[1])];
    } else {
      Typ2 = INF;
    }
    in = i + 0;
    jn = j + 0;
    kn = k - 2;
    if (isfrozen3d(in, jn, kn, dims, Frozen)) {
      Tzm2 = T[mindex3(in, jn, kn, dims[0], dims[1])];
    } else {
      Tzm2 = INF;
    }
    in = i + 0;
    jn = j + 0;
    kn = k + 2;
    if (isfrozen3d(in, jn, kn, dims, Frozen)) {
      Tzp2 = T[mindex3(in, jn, kn, dims[0], dims[1])];
    } else {
      Tzp2 = INF;
    }

    if (usecross) {
      // Tr2t1m2 = Txm2;
      // Tr2t1p2 = Txp2;
      in = i - 0;
      jn = j - 2;
      kn = k - 2;
      if (isfrozen3d(in, jn, kn, dims, Frozen)) {
        Tr2t2m2 = T[mindex3(in, jn, kn, dims[0], dims[1])];
      } else {
        Tr2t2m2 = INF;
      }
      in = i + 0;
      jn = j + 2;
      kn = k + 2;
      if (isfrozen3d(in, jn, kn, dims, Frozen)) {
        Tr2t2p2 = T[mindex3(in, jn, kn, dims[0], dims[1])];
      } else {
        Tr2t2p2 = INF;
      }
      in = i - 0;
      jn = j - 2;
      kn = k + 2;
      if (isfrozen3d(in, jn, kn, dims, Frozen)) {
        Tr2t3m2 = T[mindex3(in, jn, kn, dims[0], dims[1])];
      } else {
        Tr2t3m2 = INF;
      }
      in = i + 0;
      jn = j + 2;
      kn = k - 2;
      if (isfrozen3d(in, jn, kn, dims, Frozen)) {
        Tr2t3p2 = T[mindex3(in, jn, kn, dims[0], dims[1])];
      } else {
        Tr2t3p2 = INF;
      }
      // Tr3t1m2 = Tym2;
      // Tr3t1p2 = Typ2;
      in = i - 2;
      jn = j + 0;
      kn = k + 2;
      if (isfrozen3d(in, jn, kn, dims, Frozen)) {
        Tr3t2m2 = T[mindex3(in, jn, kn, dims[0], dims[1])];
      } else {
        Tr3t2m2 = INF;
      }
      in = i + 2;
      jn = j + 0;
      kn = k - 2;
      if (isfrozen3d(in, jn, kn, dims, Frozen)) {
        Tr3t2p2 = T[mindex3(in, jn, kn, dims[0], dims[1])];
      } else {
        Tr3t2p2 = INF;
      }
      in = i - 2;
      jn = j - 0;
      kn = k - 2;
      if (isfrozen3d(in, jn, kn, dims, Frozen)) {
        Tr3t3m2 = T[mindex3(in, jn, kn, dims[0], dims[1])];
      } else {
        Tr3t3m2 = INF;
      }
      in = i + 2;
      jn = j + 0;
      kn = k + 2;
      if (isfrozen3d(in, jn, kn, dims, Frozen)) {
        Tr3t3p2 = T[mindex3(in, jn, kn, dims[0], dims[1])];
      } else {
        Tr3t3p2 = INF;
      }
      // Tr4t1m2 = Tzm2;
      // Tr4t1p2 = Tzp2;
      in = i - 2;
      jn = j - 2;
      kn = k - 0;
      if (isfrozen3d(in, jn, kn, dims, Frozen)) {
        Tr4t2m2 = T[mindex3(in, jn, kn, dims[0], dims[1])];
      } else {
        Tr4t2m2 = INF;
      }
      in = i + 2;
      jn = j + 2;
      kn = k + 0;
      if (isfrozen3d(in, jn, kn, dims, Frozen)) {
        Tr4t2p2 = T[mindex3(in, jn, kn, dims[0], dims[1])];
      } else {
        Tr4t2p2 = INF;
      }
      in = i - 2;
      jn = j + 2;
      kn = k - 0;
      if (isfrozen3d(in, jn, kn, dims, Frozen)) {
        Tr4t3m2 = T[mindex3(in, jn, kn, dims[0], dims[1])];
      } else {
        Tr4t3m2 = INF;
      }
      in = i + 2;
      jn = j - 2;
      kn = k + 0;
      if (isfrozen3d(in, jn, kn, dims, Frozen)) {
        Tr4t3p2 = T[mindex3(in, jn, kn, dims[0], dims[1])];
      } else {
        Tr4t3p2 = INF;
      }
      // Tr5t1m2 = Tr3t3m2;
      // Tr5t1p2 = Tr3t3p2;
      in = i - 2;
      jn = j - 2;
      kn = k + 2;
      if (isfrozen3d(in, jn, kn, dims, Frozen)) {
        Tr5t2m2 = T[mindex3(in, jn, kn, dims[0], dims[1])];
      } else {
        Tr5t2m2 = INF;
      }
      in = i + 2;
      jn = j + 2;
      kn = k - 2;
      if (isfrozen3d(in, jn, kn, dims, Frozen)) {
        Tr5t2p2 = T[mindex3(in, jn, kn, dims[0], dims[1])];
      } else {
        Tr5t2p2 = INF;
      }
      in = i - 2;
      jn = j + 2;
      kn = k + 2;
      if (isfrozen3d(in, jn, kn, dims, Frozen)) {
        Tr5t3m2 = T[mindex3(in, jn, kn, dims[0], dims[1])];
      } else {
        Tr5t3m2 = INF;
      }
      in = i + 2;
      jn = j - 2;
      kn = k - 2;
      if (isfrozen3d(in, jn, kn, dims, Frozen)) {
        Tr5t3p2 = T[mindex3(in, jn, kn, dims[0], dims[1])];
      } else {
        Tr5t3p2 = INF;
      }
      // Tr6t1m2 = Tr3t2p2;
      // Tr6t1p2 = Tr3t2m2;
      in = i - 2;
      jn = j - 2;
      kn = k - 2;
      if (isfrozen3d(in, jn, kn, dims, Frozen)) {
        Tr6t2m2 = T[mindex3(in, jn, kn, dims[0], dims[1])];
      } else {
        Tr6t2m2 = INF;
      }
      in = i + 2;
      jn = j + 2;
      kn = k + 2;
      if (isfrozen3d(in, jn, kn, dims, Frozen)) {
        Tr6t2p2 = T[mindex3(in, jn, kn, dims[0], dims[1])];
      } else {
        Tr6t2p2 = INF;
      }
      in = i - 2;
      jn = j + 2;
      kn = k - 2;
      if (isfrozen3d(in, jn, kn, dims, Frozen)) {
        Tr6t3m2 = T[mindex3(in, jn, kn, dims[0], dims[1])];
      } else {
        Tr6t3m2 = INF;
      }
      in = i + 2;
      jn = j - 2;
      kn = k + 2;
      if (isfrozen3d(in, jn, kn, dims, Frozen)) {
        Tr6t3p2 = T[mindex3(in, jn, kn, dims[0], dims[1])];
      } else {
        Tr6t3p2 = INF;
      }
    }

    /*pixels with a pixeldistance 2 from the center must be */
    /*lower in value otherwise use other side or first order */

    Tm2[0] = second_derivative(Txm1, Txm2, Txp1, Txp2);
    if (IsInf(Tm2[0])) {
      Tm2[0] = 0;
    } else {
      Order[0] = 2;
    }
    Tm2[1] = second_derivative(Tym1, Tym2, Typ1, Typ2);
    if (IsInf(Tm2[1])) {
      Tm2[1] = 0;
    } else {
      Order[1] = 2;
    }
    Tm2[2] = second_derivative(Tzm1, Tzm2, Tzp1, Tzp2);
    if (IsInf(Tm2[2])) {
      Tm2[2] = 0;
    } else {
      Order[2] = 2;
    }

    if (usecross) {
      Tm2[3] = Tm2[0];
      Order[3] = Order[0];
      Tm2[4] = second_derivative(Tr2t2m1, Tr2t2m2, Tr2t2p1, Tr2t2p2);
      if (IsInf(Tm2[4])) {
        Tm2[4] = 0;
      } else {
        Order[4] = 2;
      }
      Tm2[5] = second_derivative(Tr2t3m1, Tr2t3m2, Tr2t3p1, Tr2t3p2);
      if (IsInf(Tm2[5])) {
        Tm2[5] = 0;
      } else {
        Order[5] = 2;
      }

      Tm2[6] = Tm2[1];
      Order[6] = Order[1];
      Tm2[7] = second_derivative(Tr3t2m1, Tr3t2m2, Tr3t2p1, Tr3t2p2);
      if (IsInf(Tm2[7])) {
        Tm2[7] = 0;
      } else {
        Order[7] = 2;
      }
      Tm2[8] = second_derivative(Tr3t3m1, Tr3t3m2, Tr3t3p1, Tr3t3p2);
      if (IsInf(Tm2[8])) {
        Tm2[8] = 0;
      } else {
        Order[8] = 2;
      }

      Tm2[9] = Tm2[2];
      Order[9] = Order[2];
      Tm2[10] = second_derivative(Tr4t2m1, Tr4t2m2, Tr4t2p1, Tr4t2p2);
      if (IsInf(Tm2[10])) {
        Tm2[10] = 0;
      } else {
        Order[10] = 2;
      }
      Tm2[11] = second_derivative(Tr4t3m1, Tr4t3m2, Tr4t3p1, Tr4t3p2);
      if (IsInf(Tm2[11])) {
        Tm2[11] = 0;
      } else {
        Order[11] = 2;
      }

      Tm2[12] = Tm2[8];
      Order[12] = Order[8];
      Tm2[13] = second_derivative(Tr5t2m1, Tr5t2m2, Tr5t2p1, Tr5t2p2);
      if (IsInf(Tm2[13])) {
        Tm2[13] = 0;
      } else {
        Order[13] = 2;
      }
      Tm2[14] = second_derivative(Tr5t3m1, Tr5t3m2, Tr5t3p1, Tr5t3p2);
      if (IsInf(Tm2[14])) {
        Tm2[14] = 0;
      } else {
        Order[14] = 2;
      }

      Tm2[15] = Tm2[7];
      Order[15] = Order[7];
      Tm2[16] = second_derivative(Tr6t2m1, Tr6t2m2, Tr6t2p1, Tr6t2p2);
      if (IsInf(Tm2[16])) {
        Tm2[16] = 0;
      } else {
        Order[16] = 2;
      }
      Tm2[17] = second_derivative(Tr6t3m1, Tr6t3m2, Tr6t3p1, Tr6t3p2);
      if (IsInf(Tm2[17])) {
        Tm2[17] = 0;
      } else {
        Order[17] = 2;
      }
    }
  }

  /*Calculate the distance using x and y direction */
  Coeff[0] = 0;
  Coeff[1] = 0;
  Coeff[2] = -1 / (max(pow2(Fijk), eps));

  for (t = 0; t < 3; t++) {
    switch (Order[t]) {
      case 1:
        Coeff[0] += G1[t];
        Coeff[1] += -2.0 * Tm[t] * G1[t];
        Coeff[2] += pow2(Tm[t]) * G1[t];
        break;
      case 2:
        Coeff[0] += G2[t];
        Coeff[1] += -2.0 * Tm2[t] * G2[t];
        Coeff[2] += pow2(Tm2[t]) * G2[t];
        break;
    }
  }

  roots(Coeff, ansroot);
  Tt = max(ansroot[0], ansroot[1]);

  /*Calculate the distance using the cross directions */
  if (usecross) {
    for (q = 1; q < 6; q++) {
      /* Original Equation */
      /*    Coeff[0]=0; Coeff[1]=0; Coeff[2]=-1/(max(pow2(Fijk),eps)) */
      Coeff[0] += 0;
      Coeff[1] += 0;
      Coeff[2] += -1 / (max(pow2(Fijk), eps));

      for (t = q * 3; t < ((q + 1) * 3); t++) {
        switch (Order[t]) {
          case 1:
            Coeff[0] += G1[t];
            Coeff[1] += -2.0 * Tm[t] * G1[t];
            Coeff[2] += pow2(Tm[t]) * G1[t];
            break;
          case 2:
            Coeff[0] += G2[t];
            Coeff[1] += -2.0 * Tm2[t] * G2[t];
            Coeff[2] += pow2(Tm2[t]) * G2[t];
            break;
        }
      }
      /*Select maximum root solution and minimum distance value of both stensils
       */
      if (Coeff[0] > 0) {
        roots(Coeff, ansroot);
        Tt2 = max(ansroot[0], ansroot[1]);
        Tt = min(Tt, Tt2);
      }
    }
  }

  /*Upwind condition check, current distance must be larger */
  /*then direct neighbours used in solution */
  /*(Will this ever happen?) */
  if (usecross) {
    for (q = 0; q < 18; q++) {
      if (IsFinite(Tm[q]) && (Tt < Tm[q])) {
        Tt = Tm[minarray(Tm, 18)] + (1 / (max(Fijk, eps)));
      }
    }
  } else {
    for (q = 0; q < 3; q++) {
      if (IsFinite(Tm[q]) && (Tt < Tm[q])) {
        Tt = Tm[minarray(Tm, 3)] + (1 / (max(Fijk, eps)));
      }
    }
  }

  return Tt;
}


void msfm3d(npy_double *F,             // The input speed image
            int dims[3],           // The size of the input speed image
            npy_int64 *SourcePoints,  // The source points
            int dims_sp[2],        // The size of the source point array
            bool usesecond, 
            bool usecross,
            npy_double *T,  // The output time crosing map
            npy_double *Y)  // The output euclidean image
{
  /* Current distance values */
  npy_double Tt, Ty;

  /* Matrix containing the Frozen Pixels" */
  bool* Frozen;

  /* Augmented Fast Marching (For skeletonize) */
  bool Ed = false;  // Skip the Y output for now

  /* Number of pixels in image */
  int npixels = dims[0] * dims[1] * dims[2];

  /* Neighbour list */
  int neg_free;
  int neg_pos;
  npy_double* neg_listv;
  npy_double* neg_listx;
  npy_double* neg_listy;
  npy_double* neg_listz;
  npy_double* neg_listo;

  int* listprop;
  npy_double** listval;

  /* Neighbours 6x3 */
  int ne[18] = {-1, 0, 0, 1, 0, 0, 0, -1, 0, 0, 1, 0, 0, 0, -1, 0, 0, 1};

  /* Loop variables */
  int s, w, itt, q;

  /* Current location */
  int x, y, z, i, j, k;

  /* Index */
  int IJK_index, XYZ_index, index;

  /* Pixels which are processed and have a final distance are frozen */
  Frozen = (bool*)malloc(npixels * sizeof(int));
  for (q = 0; q < npixels; q++) {
    Frozen[q] = 0;
    T[q] = -1;
  }
  if (Ed) {
    for (q = 0; q < npixels; q++) {
      Y[q] = -1;
    }
  }

  /*Free memory to store neighbours of the (segmented) region */
  neg_free = 100000;
  neg_pos = 0;

  neg_listx = (npy_double*)malloc(neg_free * sizeof(npy_double));
  neg_listy = (npy_double*)malloc(neg_free * sizeof(npy_double));
  neg_listz = (npy_double*)malloc(neg_free * sizeof(npy_double));
  if (Ed) {
    neg_listo = (npy_double*)malloc(neg_free * sizeof(npy_double));
    for (q = 0; q < neg_free; q++) {
      neg_listo[q] = 0;
    }
  }

  /* List parameters array */
  listprop = (int*)malloc(3 * sizeof(int));
  /* Make jagged list to store a maximum of 2^64 values */
  listval = (npy_double**)malloc(64 * sizeof(npy_double*));

  /* Initialize parameter list */
  initialize_list(listval, listprop);
  neg_listv = listval[listprop[1] - 1];

  /*(There are 3 pixel classes: */
  /*  - frozen (processed) */
  /*  - narrow band (boundary) (in list to check for the next pixel with
   * smallest distance) */
  /*  - far (not yet used) */
  /* set all starting points to distance zero and frozen */
  /* and add all neighbours of the starting points to narrow list */

  for (s = 0; s < dims_sp[1]; s++) {
    /*starting point */
    x = (int)SourcePoints[0 + s * 3] - 1;
    y = (int)SourcePoints[1 + s * 3] - 1;
    z = (int)SourcePoints[2 + s * 3] - 1;
    XYZ_index = mindex3(x, y, z, dims[0], dims[1]);

    Frozen[XYZ_index] = 1;
    T[XYZ_index] = 0;
    if (Ed) {
      Y[XYZ_index] = 0;
    }
  }

  for (s = 0; s < dims_sp[1]; s++) {
    /*starting point */
    x = (int)SourcePoints[0 + s * 3] - 1;
    y = (int)SourcePoints[1 + s * 3] - 1;
    z = (int)SourcePoints[2 + s * 3] - 1;

    XYZ_index = mindex3(x, y, z, dims[0], dims[1]);
    for (w = 0; w < 6; w++) {
      /*Location of neighbour */
      i = x + ne[w];
      j = y + ne[w + 6];
      k = z + ne[w + 12];

      IJK_index = mindex3(i, j, k, dims[0], dims[1]);

      /*Check if current neighbour is not yet frozen and inside the */

      /*picture */
      if (isntfrozen3d(i, j, k, dims, Frozen)) {
        Tt = (1 / (max(F[IJK_index], eps)));
        /*Update distance in neigbour list or add to neigbour list */
        if (T[IJK_index] > 0) {
          if (neg_listv[(int)T[IJK_index]] > Tt) {
            listupdate(listval, listprop, (int)T[IJK_index], Tt);
          }
        } else {
          /*If running out of memory at a new block */
          if (neg_pos >= neg_free) {
            neg_free += 100000;
            neg_listx = (npy_double*)realloc(neg_listx, neg_free * sizeof(npy_double));
            neg_listy = (npy_double*)realloc(neg_listy, neg_free * sizeof(npy_double));
            neg_listz = (npy_double*)realloc(neg_listz, neg_free * sizeof(npy_double));
            if (Ed) {
              neg_listo =
                  (npy_double*)realloc(neg_listo, neg_free * sizeof(npy_double));
            }
          }
          list_add(listval, listprop, Tt);
          neg_listv = listval[listprop[1] - 1];
          neg_listx[neg_pos] = i;
          neg_listy[neg_pos] = j;
          neg_listz[neg_pos] = k;
          T[IJK_index] = neg_pos;
          neg_pos++;
        }
      }
    }
  }

  /*Loop through all pixels of the image */
  for (itt = 0; itt < (npixels); itt++) /* */ {
    /*Get the pixel from narrow list (boundary list) with smallest */
    /*distance value and set it to current pixel location */
    index = list_minimum(listval, listprop);
    neg_listv = listval[listprop[1] - 1];
    /* Stop if pixel distance is infinite (all pixels are processed) */
    if (IsInf(neg_listv[index])) {
      break;
    }

    /*index=minarray(neg_listv, neg_pos); */
    x = (int)neg_listx[index];
    y = (int)neg_listy[index];
    z = (int)neg_listz[index];
    XYZ_index = mindex3(x, y, z, dims[0], dims[1]);
    Frozen[XYZ_index] = 1;
    T[XYZ_index] = neg_listv[index];
    if (Ed) {
      Y[XYZ_index] = neg_listo[index];
    }

    /*Remove min value by replacing it with the last value in the array */
    list_remove_replace(listval, listprop, index);
    neg_listv = listval[listprop[1] - 1];
    if (index < (neg_pos - 1)) {
      neg_listx[index] = neg_listx[neg_pos - 1];
      neg_listy[index] = neg_listy[neg_pos - 1];
      neg_listz[index] = neg_listz[neg_pos - 1];
      if (Ed) {
        neg_listo[index] = neg_listo[neg_pos - 1];
      }
      T[(int)mindex3((int)neg_listx[index], (int)neg_listy[index],
                     (int)neg_listz[index], dims[0], dims[1])] = index;
    }
    neg_pos = neg_pos - 1;

    /*Loop through all 6 neighbours of current pixel */
    for (w = 0; w < 6; w++) {
      /*Location of neighbour */
      i = x + ne[w];
      j = y + ne[w + 6];
      k = z + ne[w + 12];
      IJK_index = mindex3(i, j, k, dims[0], dims[1]);

      /*Check if current neighbour is not yet frozen and inside the */
      /*picture */
      if (isntfrozen3d(i, j, k, dims, Frozen)) {
        Tt = CalculateDistance(T, F[IJK_index], dims, i, j, k, usesecond,
                               usecross, Frozen);
        if (Ed) {
          Ty = CalculateDistance(Y, 1, dims, i, j, k, usesecond, usecross,
                                 Frozen);
        }

        /*Update distance in neigbour list or add to neigbour list */
        IJK_index = mindex3(i, j, k, dims[0], dims[1]);
        if ((T[IJK_index] > -1) && T[IJK_index] <= listprop[0]) {
          if (neg_listv[(int)T[IJK_index]] > Tt) {
            listupdate(listval, listprop, (int)T[IJK_index], Tt);
          }
        } else {
          /*If running out of memory at a new block */
          if (neg_pos >= neg_free) {
            neg_free += 100000;
            neg_listx = (npy_double*)realloc(neg_listx, neg_free * sizeof(npy_double));
            neg_listy = (npy_double*)realloc(neg_listy, neg_free * sizeof(npy_double));
            neg_listz = (npy_double*)realloc(neg_listz, neg_free * sizeof(npy_double));
            if (Ed) {
              neg_listo =
                  (npy_double*)realloc(neg_listo, neg_free * sizeof(npy_double));
            }
          }
          list_add(listval, listprop, Tt);
          neg_listv = listval[listprop[1] - 1];
          neg_listx[neg_pos] = i;
          neg_listy[neg_pos] = j;
          neg_listz[neg_pos] = k;
          if (Ed) {
            neg_listo[neg_pos] = Ty;
          }

          T[IJK_index] = neg_pos;
          neg_pos++;
        }
      }
    }
  }
  /* Free memory */
  /* Destroy parameter list */
  destroy_list(listval, listprop);
  free(neg_listx);
  free(neg_listy);
  free(neg_listz);
  free(Frozen);
}