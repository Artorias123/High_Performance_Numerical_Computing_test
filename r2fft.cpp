#include <iostream>
#include <cmath>
#include <cstdlib>
#include <immintrin.h>
#include <ctime>
using namespace std;
constexpr double const pi = atan(double(1)) * 4;
void Swap(double *x, size_t i, size_t j)
{
    double tmp = x[i];
    x[i] = x[j];
    x[j] = tmp;
}
template <int n>
struct BitReverse4
{
    bool judg[n];
    int record[n],rec[n];
    BitReverse4()
    {
        for (int i = 0; i < n; i++)
            judg[i] = 0;
        unsigned count = 0, k = n - 1;
        while (k)
        {
            k >>= 1;
            count++;
        }
        for (unsigned i = 0; i < n; i++)
        {
            if (judg[i])
                continue;
            k = 0;
            for (unsigned j = 0; j < count; j += 2)
            {
                k |= (3 << j & i) >> j;
                k <<= 2;
            }
            record[i] = k >> 2;
            rec[i]=record[i];
            rec[rec[i]]=i;
            judg[k >> 2] = !judg[k >> 2];
        }
    }
    void operator()(double *x)
    {
        if (n < 4)
            return;
        for (int i = 0; i < n; i++)
        {
            if (!judg[i])
                Swap(x, i, record[i]);
        }
    }
};
template <size_t m, size_t n>
struct mesh
{
    double *_value, *_row[m];
    mesh()
    {
        _value = (double *)_mm_malloc(2 * m * n * sizeof(double), 32);
		for(int i=0;i<2*m*n;i++) _value[i]=0;
        for(int i=0;i<m;i++) _row[i]=_value+2*n*i;
    }
    ~mesh() { _mm_free(_value); }
    double *operator[](const size_t i) { return _row[i]; }
    const double *operator[](const size_t i) const { return _row[i]; }
};
size_t log4(size_t x)
{
    size_t k = 0;
    do
    {
        x >>= 2;
        k++;
    } while (x != 1);
    return k;
}
template <size_t m, size_t n>
struct r2fft
{
    double *Wx[3], *Wy;
    BitReverse4<n> Rx;
    BitReverse4<m> Ry;

    r2fft()
    {
        int k=log4(n);
		Wx[0] = (double*)_mm_malloc((n-4)/3*sizeof(double)*2,32);
		Wx[1] = (double*)_mm_malloc((n-4)/3*sizeof(double)*2,32);
		Wx[2] = (double*)_mm_malloc((n-4)/3*sizeof(double)*2,32);
		double *r0=Wx[0],*r1=Wx[1],*r2=Wx[2],*i0=Wx[0]+(n-4)/3,*i1=Wx[1]+(n-4)/3,*i2=Wx[2]+(n-4)/3;
		for (int i = 1; i < k; i++){
			for(int j=0;j<(1 << (i * 2));j++){
				*(r0++)=cos(2 * j * pi / (1 << (i * 2+2)));
				*(r1++)=cos(2 * 2*j * pi / (1 << (i * 2+2)));
				*(r2++)=cos(2 * 3*j * pi / (1 << (i * 2+2)));
				*(i0++)=-sin(2 * j * pi / (1 << (i * 2+2)));
				*(i1++)=-sin(2 * 2*j * pi / (1 << (i * 2+2)));
				*(i2++)=-sin(2 * 3*j * pi / (1 << (i * 2+2)));
			}
		}
        Wy=(double*)malloc(m/4*3*sizeof(double)*2);
        for (int i = 0; i < m/4 * 3; i++)
		{
            Wy[i] = cos(2 * i * pi / m);
            Wy[i+m/4*3]=-sin(2 * i * pi / m);
        }
    }
    ~r2fft()
    {
        for (int i = 0; i < 3; i++)
            _mm_free(Wx[i]);
        _mm_free(Wy);
    }
    inline void fftx(mesh<m, n> &u)
    {
        double r1, r2, r3, r4;
        __m256d mr1,mr2,mr3,mr4,mi1,mi2,mi3,mi4;
        double *wr1,*wr2,*wr3,*wi1,*wi2,*wi3,*twr1,*twr2,*twr3,*twi1,*twi2,*twi3;
        int i,j,td;
        for(int k=0;k<m;k++){
            Rx(u[k]);
            td = 1;
		    wr1=Wx[0];wr2=Wx[1];wr3=Wx[2];wi1=Wx[0]+(n-4)/3;wi2=Wx[1]+(n-4)/3;wi3=Wx[2]+(n-4)/3;
	    	i = 0;
		    do
    		{
	    		r1 = u[k][i];
		    	r2 = u[k][i + 1];
			    r3 = u[k][i + 2];
    			r4 = u[k][i + 3];
	    		u[k][i]=(r1 + r2 + r3 + r4);
		    	u[k][i + 1]=(r1 - r3);
			    u[k][i + 1+n]=(-r2 + r4);
    			u[k][i + 2]=(r1 - r2 + r3 - r4);
	    		u[k][i + 3]=u[k][i + 1];
		    	u[k][i + 3+n]=-u[k][i + 1+n];
			    i += 4;
    		} while (i < n);
	    	td <<= 2;
    		do
	    	{
		    	i = 0;
			    twr1=wr1;twr2=wr2;twr3=wr3;twi1=wi1;twi2=wi2;twi3=wi3;
    			do
	    		{
		    		wr1=twr1;wr2=twr2;wr3=twr3;wi1=twi1;wi2=twi2;wi3=twi3;
			    	for (j=0; j < td; j+=4)
				    {
					    mr1 = _mm256_load_pd(&u[k][i]);
    					mi1 = _mm256_load_pd(&u[k][i+n]);
	    				mr2 = _mm256_fmsub_pd(_mm256_load_pd(&u[k][i + td]) , _mm256_load_pd(wr1) , _mm256_load_pd(&u[k][i + td+n]) * _mm256_load_pd(wi1));
		    			mi2 = _mm256_fmadd_pd(_mm256_load_pd(&u[k][i + td]) , _mm256_load_pd(wi1) , _mm256_load_pd(&u[k][i + td+n]) * _mm256_load_pd(wr1));
			    		mr3 = _mm256_fmsub_pd(_mm256_load_pd(&u[k][i + td + td]) , _mm256_load_pd(wr2) , _mm256_load_pd(&u[k][i + td + td+n]) * _mm256_load_pd(wi2));
				    	mi3 = _mm256_fmadd_pd(_mm256_load_pd(&u[k][i + td + td]) , _mm256_load_pd(wi2) , _mm256_load_pd(&u[k][i + td + td+n]) * _mm256_load_pd(wr2));
					    mr4 = _mm256_fmsub_pd(_mm256_load_pd(&u[k][i + td + td + td]) , _mm256_load_pd(wr3) , _mm256_load_pd(&u[k][i + td + td + td+n]) * _mm256_load_pd(wi3));
    					mi4 = _mm256_fmadd_pd(_mm256_load_pd(&u[k][i + td + td + td]) , _mm256_load_pd(wi3) , _mm256_load_pd(&u[k][i + td + td + td+n]) * _mm256_load_pd(wr3));
	    				_mm256_store_pd(&u[k][i],mr1 + mr2 + mr3 + mr4);
		    			_mm256_store_pd(&u[k][i+n],mi1 + mi2 + mi3 + mi4);
			    		_mm256_store_pd(&u[k][i + td],mr1 + mi2 - mr3 - mi4);
				    	_mm256_store_pd(&u[k][i + td+n],mi1 - mr2 - mi3 + mr4);
					    _mm256_store_pd(&u[k][i + td + td],mr1 - mr2 + mr3 - mr4);
    					_mm256_store_pd(&u[k][i + td + td+n],mi1 - mi2 + mi3 - mi4);
	    				_mm256_store_pd(&u[k][i + td + td + td],mr1 - mi2 - mr3 + mi4);
		    			_mm256_store_pd(&u[k][i + td + td + td+n],mi1 + mr2 - mi3 - mr4);
			    		wr1+=4;
				    	wr2+=4;
					    wr3+=4;
    					wi1+=4;
	    				wi2+=4;
		    			wi3+=4;
			    		i+=4;
				    }
				    i += td + td + td;
			    } while (i < n);
			    td <<= 2;
		    } while (td != n);
        }
    }
	inline void ffty(mesh<m,n> &u)
    {
        int tt, td , i, j;
		__m256d r1, r2, r3, r4, i1, i2, i3, i4;
		double *tu=(double*)_mm_malloc(4*m*sizeof(double)*2,32);
        for(int k=0;k<n;k+=4)
        {
            tt=n;td=1;
		    i = 0;
		    do
		    {
    			r1 = _mm256_load_pd(&u[Ry.rec[i]][k]);
	    		i1 = _mm256_load_pd(&u[Ry.rec[i]][k+m]);
		    	r2 = _mm256_load_pd(&u[Ry.rec[i+1]][k]);
			    i2 = _mm256_load_pd(&u[Ry.rec[i+1]][k+m]);
    			r3 = _mm256_load_pd(&u[Ry.rec[i+2]][k]);
	    		i3 = _mm256_load_pd(&u[Ry.rec[i+2]][k+m]);
		    	r4 = _mm256_load_pd(&u[Ry.rec[i+3]][k]);
			    i4 = _mm256_load_pd(&u[Ry.rec[i+3]][k+m]);
    			_mm256_store_pd(&tu[i*4],r1 + r2 + r3 + r4);
    			_mm256_store_pd(&tu[(i+m)*4],i1 + i2 + i3 + i4);
	    		_mm256_store_pd(&tu[(i+td)*4],r1 + i2 - r3 - i4);
				_mm256_store_pd(&tu[(i+td+m)*4],i1 - r2 - i3 + r4);
		    	_mm256_store_pd(&tu[(i+td+td)*4],r1 - r2 + r3 - r4);
			    _mm256_store_pd(&tu[(i+td+td+m)*4],i1 - i2 + i3 - i4);
    			_mm256_store_pd(&tu[(i+td+td+td)*4],r1 - i2 - r3 + i4);
	    		_mm256_store_pd(&tu[(i+td+td+td+m)*4],i1 + r2 - i3 - r4);
    			i += 4;
	    	} while (i < m);
		    tt >>= 2;
    		td <<= 2;
	    	do
		    {
    			i = 0;
	    		tt >>= 2;
		    	do
			    {
    				j = 0;
	    			r1 = _mm256_load_pd(&tu[i*4]);
		    		i1 = _mm256_load_pd(&tu[(i+m)*4]);
			    	r2 = _mm256_load_pd(&tu[(i+td)*4]);
				    i2 = _mm256_load_pd(&tu[(i+td+m)*4]);
    				r3 = _mm256_load_pd(&tu[(i+td+td)*4]);
	    			i3 = _mm256_load_pd(&tu[(i+td+td+m)*4]);
		    		r4 = _mm256_load_pd(&tu[(i+td+td+td)*4]);
			    	i4 = _mm256_load_pd(&tu[(i+td+td+td+m)*4]);
				    _mm256_store_pd(&tu[i*4],r1 + r2 + r3 + r4);
    				_mm256_store_pd(&tu[(i+m)*4],i1 + i2 + i3 + i4);
	    			_mm256_store_pd(&tu[(i+td)*4],r1 + i2 - r3 - i4);
		    		_mm256_store_pd(&tu[(i+td+m)*4],i1 - r2 - i3 + r4);
			    	_mm256_store_pd(&tu[(i+td+td)*4],r1 - r2 + r3 - r4);
				    _mm256_store_pd(&tu[(i+m+td+td)*4],i1 - i2 + i3 - i4);
    				_mm256_store_pd(&tu[(i+td+td+td)*4],r1 - i2 - r3 + i4);
	    			_mm256_store_pd(&tu[(i+m+td+td+td)*4],i1 + r2 - i3 - r4);
		    		i++;
			    	j++;
				    for (; j < td; j++)
    				{
	    				r1 = _mm256_load_pd(&tu[i*4]);
		    			i1 = _mm256_load_pd(&tu[(i+m)*4]);
			    		r2 = _mm256_fmsub_pd(_mm256_load_pd(&tu[(i+td)*4]) , _mm256_set1_pd(Wy[j * tt]) , _mm256_load_pd(&tu[(i+td+m)*4]) * _mm256_set1_pd(Wy[j * tt+m/4 * 3]));
				    	i2 = _mm256_fmadd_pd(_mm256_load_pd(&tu[(i+td)*4]) , _mm256_set1_pd(Wy[j * tt+m/4 * 3]) , _mm256_load_pd(&tu[(i+td+m)*4]) * _mm256_set1_pd(Wy[j * tt]));
					    r3 = _mm256_fmsub_pd(_mm256_load_pd(&tu[(i+td+td)*4]) , _mm256_set1_pd(Wy[j * tt + j * tt]) , _mm256_load_pd(&tu[(i+td+td+m)*4]) * _mm256_set1_pd(Wy[j * tt + j * tt+m/4 * 3]));
    					i3 = _mm256_fmadd_pd(_mm256_load_pd(&tu[(i+td+td)*4]) , _mm256_set1_pd(Wy[j * tt + j * tt+m/4 * 3]) , _mm256_load_pd(&tu[(i+td+td+m)*4]) * _mm256_set1_pd(Wy[j * tt + j * tt]));
	    				r4 = _mm256_fmsub_pd(_mm256_load_pd(&tu[(i+td+td+td)*4]) , _mm256_set1_pd(Wy[j * tt + j * tt + j * tt]) , _mm256_load_pd(&tu[(i+td+td+td+m)*4]) * _mm256_set1_pd(Wy[j * tt + j * tt + j * tt+m/4 * 3]));
		    			i4 = _mm256_fmadd_pd(_mm256_load_pd(&tu[(i+td+td+td)*4]) , _mm256_set1_pd(Wy[j * tt + j * tt + j * tt+m/4 * 3]) , _mm256_load_pd(&tu[(i+td+td+td+m)*4]) * _mm256_set1_pd(Wy[j * tt + j * tt + j * tt]));
			    		_mm256_store_pd(&tu[i*4],r1 + r2 + r3 + r4);
				    	_mm256_store_pd(&tu[(i+m)*4],i1 + i2 + i3 + i4);
					    _mm256_store_pd(&tu[(i+td)*4],r1 + i2 - r3 - i4);
    					_mm256_store_pd(&tu[(i+td+m)*4],i1 - r2 - i3 + r4);
	    				_mm256_store_pd(&tu[(i+td+td)*4],r1 - r2 + r3 - r4);
		    			_mm256_store_pd(&tu[(i+td+td+m)*4],i1 - i2 + i3 - i4);
			    		_mm256_store_pd(&tu[(i+td+td+td)*4],r1 - i2 - r3 + i4);
				    	_mm256_store_pd(&tu[(i+td+td+td+m)*4],i1 + r2 - i3 - r4);
					    i++;
    				}
	    			i += td + td + td;
		    	} while (i < m);
			    td <<= 2;
    		} while (tt != 4);
			i = 0;
	    	tt >>= 2;
    		j = 0;
    		r1 = _mm256_load_pd(&tu[i*4]);
		    i1 = _mm256_load_pd(&tu[(i+m)*4]);
			r2 = _mm256_load_pd(&tu[(i+td)*4]);
		    i2 = _mm256_load_pd(&tu[(i+td+m)*4]);
			r3 = _mm256_load_pd(&tu[(i+td+td)*4]);
			i3 = _mm256_load_pd(&tu[(i+td+td+m)*4]);
	    	r4 = _mm256_load_pd(&tu[(i+td+td+td)*4]);
	    	i4 = _mm256_load_pd(&tu[(i+td+td+td+m)*4]);
		    _mm256_store_pd(&u[i][k],r1 + r2 + r3 + r4);
	    	_mm256_store_pd(&u[i][k+m],i1 + i2 + i3 + i4);
		    _mm256_store_pd(&u[i + 1][k],r1 + i2 - r3 - i4);
			_mm256_store_pd(&u[i + 1][k+m],i1 - r2 - i3 + r4);
    		_mm256_store_pd(&u[i + 2][k],r1 - r2 + r3 - r4);
	    	_mm256_store_pd(&u[i + 2][k+m],i1 - i2 + i3 - i4);
		    _mm256_store_pd(&u[i + 3][k],r1 - i2 - r3 + i4);
			_mm256_store_pd(&u[i + 3][k+m],i1 + r2 - i3 - r4);
			i++;
		    j++;
			for (; j < td; j++)
    		{
				r1 = _mm256_load_pd(&tu[i*4]);
    			i1 = _mm256_load_pd(&tu[(i+m)*4]);
		    	r2 = _mm256_fmsub_pd(_mm256_load_pd(&tu[(i+td)*4]) , _mm256_set1_pd(Wy[j * tt]) , _mm256_load_pd(&tu[(i+td+m)*4]) * _mm256_set1_pd(Wy[j * tt+m/4 * 3]));
			    i2 = _mm256_fmadd_pd(_mm256_load_pd(&tu[(i+td)*4]) , _mm256_set1_pd(Wy[j * tt+m/4 * 3]) , _mm256_load_pd(&tu[(i+td+m)*4]) * _mm256_set1_pd(Wy[j * tt]));
			    r3 = _mm256_fmsub_pd(_mm256_load_pd(&tu[(i+td+td)*4]) , _mm256_set1_pd(Wy[j * tt + j * tt]) , _mm256_load_pd(&tu[(i+td+td+m)*4]) * _mm256_set1_pd(Wy[j * tt + j * tt+m/4 * 3]));
    			i3 = _mm256_fmadd_pd(_mm256_load_pd(&tu[(i+td+td)*4]) , _mm256_set1_pd(Wy[j * tt + j * tt+m/4 * 3]) , _mm256_load_pd(&tu[(i+td+td+m)*4]) * _mm256_set1_pd(Wy[j * tt + j * tt]));
    			r4 = _mm256_fmsub_pd(_mm256_load_pd(&tu[(i+td+td+td)*4]) , _mm256_set1_pd(Wy[j * tt + j * tt + j * tt]) , _mm256_load_pd(&tu[(i+td+td+td+m)*4]) * _mm256_set1_pd(Wy[j * tt + j * tt + j * tt+m/4 * 3]));
	    		i4 = _mm256_fmadd_pd(_mm256_load_pd(&tu[(i+td+td+td)*4]) , _mm256_set1_pd(Wy[j * tt + j * tt + j * tt+m/4 * 3]) , _mm256_load_pd(&tu[(i+td+td+td+m)*4]) * _mm256_set1_pd(Wy[j * tt + j * tt + j * tt]));
		    	_mm256_store_pd(&u[i][k],r1 + r2 + r3 + r4);
				_mm256_store_pd(&u[i][k+m],i1 + i2 + i3 + i4);
			    _mm256_store_pd(&u[i + td][k],r1 + i2 - r3 - i4);
				_mm256_store_pd(&u[i + td][k+m],i1 - r2 - i3 + r4);
	    		_mm256_store_pd(&u[i + td + td][k],r1 - r2 + r3 - r4);
				_mm256_store_pd(&u[i + td + td][k+m],i1 - i2 + i3 - i4);
	    		_mm256_store_pd(&u[i + td + td + td][k],r1 - i2 - r3 + i4);
				_mm256_store_pd(&u[i + td + td + td][k+m],i1 + r2 - i3 - r4);
				i++;
    		}
        }
		_mm_free(tu);
    }
	inline void operator()(mesh<m,n> &u)
	{
		fftx(u);
		ffty(u);
	}
	/*	
    void ffty(mesh<m,n> &u)
    {
        Ry(u._row);
        int tt, td , i, j;
		__m256d r1, r2, r3, r4, i1, i2, i3, i4;
        for(int k=0;k<n;k+=4)
        {
            tt=n;td=1;
		    i = 0;
		    do
		    {
    			r1 = _mm256_load_pd(&u[i][k]);
	    		i1 = _mm256_load_pd(&u[i][k+m]);
		    	r2 = _mm256_load_pd(&u[i + 1][k]);
			    i2 = _mm256_load_pd(&u[i + 1][k+m]);
    			r3 = _mm256_load_pd(&u[i + 2][k]);
	    		i3 = _mm256_load_pd(&u[i + 2][k+m]);
		    	r4 = _mm256_load_pd(&u[i + 3][k]);
			    i4 = _mm256_load_pd(&u[i + 3][k+m]);
    			_mm256_store_pd(&u[i][k],r1 + r2 + r3 + r4);
	    		_mm256_store_pd(&u[i][k+m],i1 + i2 + i3 + i4);
		    	_mm256_store_pd(&u[i + 1][k],r1 + i2 - r3 - i4);
			    _mm256_store_pd(&u[i + 1][k+m],i1 - r2 - i3 + r4);
    			_mm256_store_pd(&u[i + 2][k],r1 - r2 + r3 - r4);
	    		_mm256_store_pd(&u[i + 2][k+m],i1 - i2 + i3 - i4);
		    	_mm256_store_pd(&u[i + 3][k],r1 - i2 - r3 + i4);
			    _mm256_store_pd(&u[i + 3][k+m],i1 + r2 - i3 - r4);
    			i += 4;
	    	} while (i < m);
		    tt >>= 2;
    		td <<= 2;
	    	do
		    {
    			i = 0;
	    		tt >>= 2;
		    	do
			    {
    				j = 0;
	    			r1 = _mm256_load_pd(&u[i][k]);
		    		i1 = _mm256_load_pd(&u[i][k+m]);
			    	r2 = _mm256_load_pd(&u[i + td][k]);
				    i2 = _mm256_load_pd(&u[i + td][k+m]);
    				r3 = _mm256_load_pd(&u[i + td + td][k]);
	    			i3 = _mm256_load_pd(&u[i + td + td][k+m]);
		    		r4 = _mm256_load_pd(&u[i + td + td + td][k]);
			    	i4 = _mm256_load_pd(&u[i + td + td + td][k+m]);
				    _mm256_store_pd(&u[i][k],r1 + r2 + r3 + r4);
    				_mm256_store_pd(&u[i][k+m],i1 + i2 + i3 + i4);
	    			_mm256_store_pd(&u[i + td][k],r1 + i2 - r3 - i4);
		    		_mm256_store_pd(&u[i + td][k+m],i1 - r2 - i3 + r4);
			    	_mm256_store_pd(&u[i + td + td][k],r1 - r2 + r3 - r4);
				    _mm256_store_pd(&u[i + td + td][k+m],i1 - i2 + i3 - i4);
    				_mm256_store_pd(&u[i + td + td + td][k],r1 - i2 - r3 + i4);
	    			_mm256_store_pd(&u[i + td + td + td][k+m],i1 + r2 - i3 - r4);
		    		i++;
			    	j++;
				    for (; j < td; j++)
    				{
	    				r1 = _mm256_load_pd(&u[i][k]);
		    			i1 = _mm256_load_pd(&u[i][k+m]);
			    		r2 = _mm256_fmsub_pd(_mm256_load_pd(&u[i + td][k]) , _mm256_set1_pd(Wy[j * tt]) , _mm256_load_pd(&u[i + td][k+m]) * _mm256_set1_pd(Wy[j * tt+m/4 * 3]));
				    	i2 = _mm256_fmadd_pd(_mm256_load_pd(&u[i + td][k]) , _mm256_set1_pd(Wy[j * tt+m/4 * 3]) , _mm256_load_pd(&u[i + td][k+m]) * _mm256_set1_pd(Wy[j * tt]));
					    r3 = _mm256_fmsub_pd(_mm256_load_pd(&u[i + td + td][k]) , _mm256_set1_pd(Wy[j * tt + j * tt]) , _mm256_load_pd(&u[i + td + td][k+m]) * _mm256_set1_pd(Wy[j * tt + j * tt+m/4 * 3]));
    					i3 = _mm256_fmadd_pd(_mm256_load_pd(&u[i + td + td][k]) , _mm256_set1_pd(Wy[j * tt + j * tt+m/4 * 3]) , _mm256_load_pd(&u[i + td + td][k+m]) * _mm256_set1_pd(Wy[j * tt + j * tt]));
	    				r4 = _mm256_fmsub_pd(_mm256_load_pd(&u[i + td + td + td][k]) , _mm256_set1_pd(Wy[j * tt + j * tt + j * tt]) , _mm256_load_pd(&u[i + td + td + td][k+m]) * _mm256_set1_pd(Wy[j * tt + j * tt + j * tt+m/4 * 3]));
		    			i4 = _mm256_fmadd_pd(_mm256_load_pd(&u[i + td + td + td][k]) , _mm256_set1_pd(Wy[j * tt + j * tt + j * tt+m/4 * 3]) , _mm256_load_pd(&u[i + td + td + td][k+m]) * _mm256_set1_pd(Wy[j * tt + j * tt + j * tt]));
			    		_mm256_store_pd(&u[i][k],r1 + r2 + r3 + r4);
				    	_mm256_store_pd(&u[i][k+m],i1 + i2 + i3 + i4);
					    _mm256_store_pd(&u[i + td][k],r1 + i2 - r3 - i4);
    					_mm256_store_pd(&u[i + td][k+m],i1 - r2 - i3 + r4);
	    				_mm256_store_pd(&u[i + td + td][k],r1 - r2 + r3 - r4);
		    			_mm256_store_pd(&u[i + td + td][k+m],i1 - i2 + i3 - i4);
			    		_mm256_store_pd(&u[i + td + td + td][k],r1 - i2 - r3 + i4);
				    	_mm256_store_pd(&u[i + td + td + td][k+m],i1 + r2 - i3 - r4);
					    i++;
    				}
	    			i += td + td + td;
		    	} while (i < m);
			    td <<= 2;
    		} while (tt != 1);
        }
    }
	*/
};
#define N 256
int main()
{
    mesh<N,N> x;
    r2fft<N,N> fft;
    for(int i=0;i<N;i++){
        for(int j=0;j<N;j++) x[i][j]=j; 
        //x[i][j]=double(rand())/RAND_MAX;
    }
    clock_t s=clock();
    fft(x);
	/*
    for(int i=0;i<N;i++){
        for(int j=0;j<N;j++) cout<<x[i][j]<<","<<x[i][j+N]<<'\t';
        cout<<endl;
    }
	*/
	/*
	for(int j=0;j<N;j++) cout<<x[0][j]<<","<<x[0][j+N]<<'\t';
	cout<<endl;
	*/
    cout<<"time : "<<static_cast<double>(clock()-s)/CLOCKS_PER_SEC<<endl;
    return 0;
}