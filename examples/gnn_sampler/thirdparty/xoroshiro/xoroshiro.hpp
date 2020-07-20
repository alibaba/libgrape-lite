#ifndef XOROSHIRO_HPP_INCLUDED
#define XOROSHIRO_HPP_INCLUDED 1

/*
 * A C++ implementation of a family of Xoroshiro generators.
 *
 * See:
 *     https://en.wikipedia.org/wiki/Xoroshiro128%2B
 *     http://xoroshiro.di.unimi.it/xoroshiro128plus.c
 *
 * Based on the design and public domain implementation of
 * Xoroshiro128+ by written in 2016 by David Blackman and Sebastiano
 * Vigna.  Also updated for the 2018 V1.0 versions of Xoroshiro.
 *
 * Xoroshiro128+ fails PractRand and TestU01 because of weak low order
 * bits.  Blackman and Vigna recommend users take care to avoid
 * relying on the low bits.  See their original source for details.
 * My test results are included below.
 *
 * Xoroshiro128+ is a variant on the ideas given by George Marsaglia
 * for XorShift, and as such it is possible to use the same
 * binary-matrix techniques to create the necessary magic constants to
 * define other members of the Xoroshiro family at different bit
 * sizes.  In particular, Xoroshiro64+ may be sufficient for some
 * applications and will run better on some 32-bit machines.
 *
 * The MIT License (MIT)
 *
 * Copyright (c) 2017-19 Melissa E. O'Neill
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */


#include <cstdint>

namespace xoroshiro_detail {

template <typename itype, typename rtype,
          unsigned int a, unsigned int b, unsigned int c>
class xoroshiro {
protected:
    itype s0_, s1_;

    static constexpr unsigned int ITYPE_BITS = 8*sizeof(itype);
    static constexpr unsigned int RTYPE_BITS = 8*sizeof(rtype);
    
    static inline itype rotl(const itype x, int k) {
        return (x << k) | (x >> (ITYPE_BITS - k));
    }

public:
    using result_type = rtype;

    static constexpr result_type min() { return 0; }
    static constexpr result_type max() { return ~ result_type(0); }

    xoroshiro(itype s0 = itype(0xc1f651c67c62c6e0),
              itype s1 = itype(0x30d89576f866ac9f))
        // Easter-egg seed value for Xoroshiro128+ to remind users that
        // they should seed their PRNGs properly.
        : s0_(s0), s1_((s0 || s1) ? s1 : 1)
    {
            // Nothing (else) to do.
    }
              
    void advance()
    {
        s1_ ^= s0_;
        s0_ = rotl(s0_, a) ^ s1_ ^ (s1_ << b);
        s1_ = rotl(s1_, c);
    }

    bool operator==(const xoroshiro& rhs)
    {
        return (s0_ == rhs.s0_) && (s1_ == rhs.s1_);
    }

    bool operator!=(const xoroshiro& rhs)
    {
        return !operator==(rhs);
    }

    // Not (yet) implemented:
    //   - arbitrary jumpahead (doable, but annoying to write).
    //   - I/O
    //   - Seeding from a seed_seq.
};

template <typename itype, typename rtype,
          unsigned int a, unsigned int b, unsigned int c>
class xoroshiro_plus : public xoroshiro<itype,rtype,a,b,c> {
private:
    using base = xoroshiro<itype,rtype,a,b,c>;
public:
    using base::base;

    rtype operator()()
    {
        const itype result = base::s0_ + base::s1_;

        base::advance();

        return result >> (base::ITYPE_BITS - base::RTYPE_BITS);
    }
};

template <typename itype, typename rtype,
          unsigned int a, unsigned int b, unsigned int c, itype mult>
class xoroshiro_star : public xoroshiro<itype,rtype,a,b,c> {
private:
    using base = xoroshiro<itype,rtype,a,b,c>;
public:
    using base::base;

    rtype operator()()
    {
        const itype result_star = base::s0_ * mult;

        base::advance();

        return result_star >> (base::ITYPE_BITS - base::RTYPE_BITS);
    }
};

template <typename itype, typename rtype,
          unsigned int a, unsigned int b, unsigned int c,
          itype mult1, unsigned int orot, itype mult2>
class xoroshiro_starstar : public xoroshiro<itype,rtype,a,b,c> {
private:
    using base = xoroshiro<itype,rtype,a,b,c>;
public:
    using base::base;

    rtype operator()()
    {
        const itype result_ss = base::rotl(base::s0_ * mult1, orot) * mult2;

        base::advance();

        return result_ss >> (base::ITYPE_BITS - base::RTYPE_BITS);
    }
};

} // namespace xoroshiro_detail
    
// 128 //

// This is the generator recommended by Vigna and Blackman.  It fails
// PractRand and TestU01 although it can be considered as passing if
// the failures are ignored. (??!?!)
//
// TestU01 linear complexity results:
//
//| Testing Xoroshiro128+ (Vigna's Code) [Low bits] [Reversed]:
//| - seed_data[2] = { 0xda1ff5a1e44ba0f8, 0x511a1a4da9858f9a}
//| ***********************************************************
//| Xoroshiro128+ (Vigna's Code) [Low bits] [Reversed]
//| scomp_LinearComp test:
//| -----------------------------------------------
//|    N =  1,  n = 250,  r =  0,    s = 1
//| -----------------------------------------------
//| Number of degrees of freedom          :    1
//| Chi2 statistic for size of jumps      :    2.32
//| p-value of test                       :    0.13
//| -----------------------------------------------
//| Normal statistic for number of jumps  :   -0.15
//| p-value of test                       :    0.56
//| -----------------------------------------------
//| CPU time used                    :  00:00:00.00
//| ***********************************************************
//| Xoroshiro128+ (Vigna's Code) [Low bits] [Reversed]
//| scomp_LinearComp test:
//| -----------------------------------------------
//|    N =  1,  n = 500,  r =  0,    s = 1
//| -----------------------------------------------
//| Number of degrees of freedom          :    2
//| Chi2 statistic for size of jumps      :    1.97
//| p-value of test                       :    0.37
//| -----------------------------------------------
//| Normal statistic for number of jumps  :   -7.27
//| p-value of test                       : 1 - 1.9e-13    *****
//| -----------------------------------------------
//| CPU time used                    :  00:00:00.00
//| ***********************************************************
//| Xoroshiro128+ (Vigna's Code) [Low bits] [Reversed]
//| scomp_LinearComp test:
//| -----------------------------------------------
//|    N =  1,  n = 1000,  r =  0,    s = 1
//| -----------------------------------------------
//| Number of degrees of freedom          :    3
//| Chi2 statistic for size of jumps      :    4.33
//| p-value of test                       :    0.23
//| -----------------------------------------------
//| Normal statistic for number of jumps  :  -17.31
//| p-value of test                       : 1 - eps1    *****
//| -----------------------------------------------
//| CPU time used                    :  00:00:00.00
//| ***********************************************************
//
// TestU01 Crush Results:
//| ========= Summary results of Crush =========
//| 
//|  Version:          TestU01 1.2.3
//|  Generator:        Xoroshiro128+ (Vigna's Code) [Low bits] [Reversed]
//|  Number of statistics:  144
//|  Total CPU time:   00:39:57.80
//|  The following tests gave p-values outside [0.001, 0.9990]:
//|  (eps  means a value < 1.0e-300):
//|  (eps1 means a value < 1.0e-15):
//| 
//|        Test                          p-value
//|  ----------------------------------------------
//|  71  LinearComp, r = 0              1 - eps1
//|  ----------------------------------------------
//|  All other tests were passed
//
// TestU01 BigCrush Results:
//
//| ========= Summary results of BigCrush =========
//| 
//|  Version:          TestU01 1.2.3
//|  Generator:        Xoroshiro128+ (Vigna's Code) [Low bits] [Reversed]
//|  Number of statistics:  160
//|  Total CPU time:   04:14:51.79
//|  The following tests gave p-values outside [0.001, 0.9990]:
//|  (eps  means a value < 1.0e-300):
//|  (eps1 means a value < 1.0e-15):
//| 
//|        Test                          p-value
//|  ----------------------------------------------
//|  68  MatrixRank, L=1000, r=0          eps  
//|  71  MatrixRank, L=5000               eps  
//|  80  LinearComp, r = 0              1 - eps1
//|  ----------------------------------------------
//|  All other tests were passed
//
// PractRand Results (standard: RNG_test stdin64 -tlmin 23):
//
//| RNG_test using PractRand version 0.93
//| RNG = RNG_stdin64, seed = 0x7ddef6ce
//| test set = normal, folding = standard (64 bit)
//| 
//| rng=RNG_stdin64, seed=0x7ddef6ce
//| length= 8 megabytes (2^23 bytes), time= 0.2 seconds
//|   no anomalies in 106 test result(s)
//| 
//| rng=RNG_stdin64, seed=0x7ddef6ce
//| length= 16 megabytes (2^24 bytes), time= 1.2 seconds
//|   no anomalies in 116 test result(s)
//| 
//| rng=RNG_stdin64, seed=0x7ddef6ce
//| length= 32 megabytes (2^25 bytes), time= 2.3 seconds
//|   Test Name                         Raw       Processed     Evaluation
//|   [Low1/64]BRank(12):256(1)         R= +2650  p~=  9.8e-799   FAIL !!!!!!!   
//|   ...and 126 test result(s) without anomalies
//
// PractRand (extended: RNG_test stdin64 -tlmaxonly -te 1 -tf 2)
//| RNG = RNG_stdin64, seed = 0x88ab2def
//| test set = expanded, folding = extra
//| 
//| rng=RNG_stdin64, seed=0x88ab2def
//| length= 32 megabytes (2^25 bytes), time= 2.3 seconds
//|   Test Name                         Raw       Processed     Evaluation
//|   [Low1/16]BRank(18):768(1)         R= +1272  p~=  5.4e-384   FAIL !!!!!!!
//|   [Low1/32]BRank(18):384(1)         R= +1272  p~=  5.4e-384   FAIL !!!!!!!
//|   [Low1/32]BRank(18):512(1)         R= +2650  p~=  9.8e-799   FAIL !!!!!!!
//|   [Low1/64]BRank(18):256(2)         R= +3748  p~=  3e-1129    FAIL !!!!!!!!
//|   [Low1/64]BRank(18):384(1)         R= +5405  p~=  3e-1628    FAIL !!!!!!!!
//|   [Low4/64]BRank(18):768(1)         R= +1272  p~=  5.4e-384   FAIL !!!!!!!
//|   ...and 790 test result(s) without anomalies
//:
//:
//| rng=RNG_stdin64, seed=0x88ab2def
//| length= 8 terabytes (2^43 bytes), time= 519110 seconds
//|   Test Name                         Raw       Processed     Evaluation
//|   BRank(18):12K(8)                  R= +3598  p~=  2e-2310    FAIL !!!!!!!!
//|   BRank(18):16K(32)                 R=+14990  p~= 0           FAIL !!!!!!!!
//|   BRank(18):24K(8)                  R=+15289  p~=  4e-9814    FAIL !!!!!!!!
//|   BRank(18):32K(16)                 R=+32644  p~= 0           FAIL !!!!!!!!
//|   BRank(18):48K(4)                  R=+27344  p~= 0           FAIL !!!!!!!!
//|   BRank(18):64K(11)                 R=+63622  p~= 0           FAIL !!!!!!!!
//|   DC6-5x4Bytes-1                    R=  +6.5  p =  3.4e-4   unusual
//|   [Low1/8]BRank(18):1536(4)         R= +2544  p~=  4e-1354    FAIL !!!!!!!!
//|   [Low1/8]BRank(18):2K(8)           R= +7495  p~=  1e-4811    FAIL !!!!!!!!
//|   [Low1/8]BRank(18):3K(2)           R= +7644  p~=  3e-2302    FAIL !!!!!!!!
//|   [Low1/8]BRank(18):4K(8)           R=+23082  p~= 0           FAIL !!!!!!!!
//|   [Low1/8]BRank(18):6K(2)           R=+19335  p~=  2e-5821    FAIL !!!!!!!!
//|   [Low1/8]BRank(18):8K(4)           R=+38366  p~= 0           FAIL !!!!!!!!
//|   [Low1/8]BRank(18):12K(1)          R=+30205  p~=  1e-9093    FAIL !!!!!!!!
//|   [Low1/8]BRank(18):16K(4)          R=+82453  p~= 0           FAIL !!!!!!!!
//|   [Low1/8]BRank(18):24K(1)          R=+63271  p~= 0           FAIL !!!!!!!!
//|   [Low1/8]BRank(18):32K(2)          R=+120653 p~= 0           FAIL !!!!!!!!
//|   [Low1/8]BRank(18):48K(1)          R=+129402 p~= 0           FAIL !!!!!!!!
//|   [Low1/8]BRank(18):64K(1)          R=+173490 p~= 0           FAIL !!!!!!!!
//|   [Low1/16]BRank(18):768(4)         R= +2544  p~=  4e-1354    FAIL !!!!!!!!
//|   [Low1/16]BRank(18):1K(8)          R= +7495  p~=  1e-4811    FAIL !!!!!!!!
//|   [Low1/16]BRank(18):1536(2)        R= +7644  p~=  3e-2302    FAIL !!!!!!!!
//|   [Low1/16]BRank(18):2K(8)          R=+23082  p~= 0           FAIL !!!!!!!!
//|   [Low1/16]BRank(18):3K(2)          R=+19335  p~=  2e-5821    FAIL !!!!!!!!
//|   [Low1/16]BRank(18):4K(4)          R=+38366  p~= 0           FAIL !!!!!!!!
//|   [Low1/16]BRank(18):6K(1)          R=+30205  p~=  1e-9093    FAIL !!!!!!!!
//|   [Low1/16]BRank(18):8K(4)          R=+82453  p~= 0           FAIL !!!!!!!!
//|   [Low1/16]BRank(18):12K(1)         R=+63271  p~= 0           FAIL !!!!!!!!
//|   [Low1/16]BRank(18):16K(2)         R=+120653 p~= 0           FAIL !!!!!!!!
//|   [Low1/16]BRank(18):24K(1)         R=+129402 p~= 0           FAIL !!!!!!!!
//|   [Low1/16]BRank(18):32K(2)         R=+245352 p~= 0           FAIL !!!!!!!!
//|   [Low1/16]BRank(18):48K(1)         R=+261665 p~= 0           FAIL !!!!!!!!
//|   [Low1/32]BRank(18):384(4)         R= +2544  p~=  4e-1354    FAIL !!!!!!!!
//|   [Low1/32]BRank(18):512(8)         R= +7495  p~=  1e-4811    FAIL !!!!!!!!
//|   [Low1/32]BRank(18):768(4)         R=+10811  p~=  1e-5750    FAIL !!!!!!!!
//|   [Low1/32]BRank(18):1K(8)          R=+23082  p~= 0           FAIL !!!!!!!!
//|   [Low1/32]BRank(18):1536(2)        R=+19335  p~=  2e-5821    FAIL !!!!!!!!
//|   [Low1/32]BRank(18):2K(4)          R=+38366  p~= 0           FAIL !!!!!!!!
//|   [Low1/32]BRank(18):3K(2)          R=+42716  p~= 0           FAIL !!!!!!!!
//|   [Low1/32]BRank(18):4K(4)          R=+82453  p~= 0           FAIL !!!!!!!!
//|   [Low1/32]BRank(18):6K(1)          R=+63271  p~= 0           FAIL !!!!!!!!
//|   [Low1/32]BRank(18):8K(2)          R=+120653 p~= 0           FAIL !!!!!!!!
//|   [Low1/32]BRank(18):12K(1)         R=+129402 p~= 0           FAIL !!!!!!!!
//|   [Low1/32]BRank(18):16K(2)         R=+245352 p~= 0           FAIL !!!!!!!!
//|   [Low1/32]BRank(18):24K(1)         R=+261665 p~= 0           FAIL !!!!!!!!
//|   [Low1/32]BRank(18):32K(1)         R=+349841 p~= 0           FAIL !!!!!!!!
//|   [Low1/64]BRank(18):256(16)        R=+10600  p~=  3e-7802    FAIL !!!!!!!!
//|   [Low1/64]BRank(18):384(4)         R=+10811  p~=  1e-5750    FAIL !!!!!!!!
//|   [Low1/64]BRank(18):512(8)         R=+23082  p~= 0           FAIL !!!!!!!!
//|   [Low1/64]BRank(18):768(4)         R=+27344  p~= 0           FAIL !!!!!!!!
//|   [Low1/64]BRank(18):1K(8)          R=+54257  p~= 0           FAIL !!!!!!!!
//|   [Low1/64]BRank(18):1536(2)        R=+42716  p~= 0           FAIL !!!!!!!!
//|   [Low1/64]BRank(18):2K(4)          R=+82453  p~= 0           FAIL !!!!!!!!
//|   [Low1/64]BRank(18):3K(2)          R=+89478  p~= 0           FAIL !!!!!!!!
//|   [Low1/64]BRank(18):4K(4)          R=+170629 p~= 0           FAIL !!!!!!!!
//|   [Low1/64]BRank(18):6K(1)          R=+129402 p~= 0           FAIL !!!!!!!!
//|   [Low1/64]BRank(18):8K(2)          R=+245352 p~= 0           FAIL !!!!!!!!
//|   [Low1/64]BRank(18):12K(1)         R=+261665 p~= 0           FAIL !!!!!!!!
//|   [Low1/64]BRank(18):16K(2)         R=+494750 p~= 0           FAIL !!!!!!!!
//|   [Low1/64]BRank(18):24K(1)         R=+526192 p~= 0           FAIL !!!!!!!!
//|   [Low1/64]BRank(18):32K(1)         R=+702543 p~= 0           FAIL !!!!!!!!
//|   [Low4/16]BRank(18):3K(4)          R= +2544  p~=  4e-1354    FAIL !!!!!!!!
//|   [Low4/16]BRank(18):4K(16)         R=+10600  p~=  3e-7802    FAIL !!!!!!!!
//|   [Low4/16]BRank(18):6K(4)          R=+10811  p~=  1e-5750    FAIL !!!!!!!!
//|   [Low4/16]BRank(18):8K(8)          R=+23082  p~= 0           FAIL !!!!!!!!
//|   [Low4/16]BRank(18):12K(2)         R=+19335  p~=  2e-5821    FAIL !!!!!!!!
//|   [Low4/16]BRank(18):16K(8)         R=+54257  p~= 0           FAIL !!!!!!!!
//|   [Low4/16]BRank(18):24K(2)         R=+42716  p~= 0           FAIL !!!!!!!!
//|   [Low4/16]BRank(18):32K(4)         R=+82453  p~= 0           FAIL !!!!!!!!
//|   [Low4/16]BRank(18):48K(1)         R=+63271  p~= 0           FAIL !!!!!!!!
//|   [Low4/16]BRank(18):64K(2)         R=+120653 p~= 0           FAIL !!!!!!!!
//|   [Low4/32]BRank(18):1536(4)        R= +2544  p~=  4e-1354    FAIL !!!!!!!!
//|   [Low4/32]BRank(18):2K(8)          R= +7495  p~=  1e-4811    FAIL !!!!!!!!
//|   [Low4/32]BRank(18):3K(2)          R= +7644  p~=  3e-2302    FAIL !!!!!!!!
//|   [Low4/32]BRank(18):4K(8)          R=+23082  p~= 0           FAIL !!!!!!!!
//|   [Low4/32]BRank(18):6K(2)          R=+19335  p~=  2e-5821    FAIL !!!!!!!!
//|   [Low4/32]BRank(18):8K(4)          R=+38366  p~= 0           FAIL !!!!!!!!
//|   [Low4/32]BRank(18):12K(1)         R=+30205  p~=  1e-9093    FAIL !!!!!!!!
//|   [Low4/32]BRank(18):16K(4)         R=+82453  p~= 0           FAIL !!!!!!!!
//|   [Low4/32]BRank(18):24K(1)         R=+63271  p~= 0           FAIL !!!!!!!!
//|   [Low4/32]BRank(18):32K(2)         R=+120653 p~= 0           FAIL !!!!!!!!
//|   [Low4/32]BRank(18):48K(1)         R=+129402 p~= 0           FAIL !!!!!!!!
//|   [Low4/32]BRank(18):64K(1)         R=+174868 p~= 0           FAIL !!!!!!!!
//|   [Low4/64]BRank(18):768(4)         R= +2544  p~=  4e-1354    FAIL !!!!!!!!
//|   [Low4/64]BRank(18):1K(8)          R= +7495  p~=  1e-4811    FAIL !!!!!!!!
//|   [Low4/64]BRank(18):1536(2)        R= +7644  p~=  3e-2302    FAIL !!!!!!!!
//|   [Low4/64]BRank(18):2K(8)          R=+23082  p~= 0           FAIL !!!!!!!!
//|   [Low4/64]BRank(18):3K(2)          R=+19335  p~=  2e-5821    FAIL !!!!!!!!
//|   [Low4/64]BRank(18):4K(4)          R=+38366  p~= 0           FAIL !!!!!!!!
//|   [Low4/64]BRank(18):6K(1)          R=+30205  p~=  1e-9093    FAIL !!!!!!!!
//|   [Low4/64]BRank(18):8K(4)          R=+82453  p~= 0           FAIL !!!!!!!!
//|   [Low4/64]BRank(18):12K(1)         R=+63271  p~= 0           FAIL !!!!!!!!
//|   [Low4/64]BRank(18):16K(2)         R=+120653 p~= 0           FAIL !!!!!!!!
//|   [Low4/64]BRank(18):24K(1)         R=+129402 p~= 0           FAIL !!!!!!!!
//|   [Low4/64]BRank(18):32K(2)         R=+247300 p~= 0           FAIL !!!!!!!!
//|   [Low4/64]BRank(18):48K(1)         R=+351219 p~= 0           FAIL !!!!!!!!
//|   [Low8/32]BRank(18):3K(4)          R= +2544  p~=  4e-1354    FAIL !!!!!!!!
//|   [Low8/32]BRank(18):4K(16)         R=+10600  p~=  3e-7802    FAIL !!!!!!!!
//|   [Low8/32]BRank(18):6K(4)          R=+10811  p~=  1e-5750    FAIL !!!!!!!!
//|   [Low8/32]BRank(18):8K(8)          R=+23082  p~= 0           FAIL !!!!!!!!
//|   [Low8/32]BRank(18):12K(2)         R=+19335  p~=  2e-5821    FAIL !!!!!!!!
//|   [Low8/32]BRank(18):16K(8)         R=+54257  p~= 0           FAIL !!!!!!!!
//|   [Low8/32]BRank(18):24K(2)         R=+42716  p~= 0           FAIL !!!!!!!!
//|   [Low8/32]BRank(18):32K(4)         R=+82453  p~= 0           FAIL !!!!!!!!
//|   [Low8/32]BRank(18):48K(1)         R=+63271  p~= 0           FAIL !!!!!!!!
//|   [Low8/32]BRank(18):64K(2)         R=+120653 p~= 0           FAIL !!!!!!!!
//|   [Low8/64]BRank(18):1536(4)        R= +2544  p~=  4e-1354    FAIL !!!!!!!!
//|   [Low8/64]BRank(18):2K(8)          R= +7495  p~=  1e-4811    FAIL !!!!!!!!
//|   [Low8/64]BRank(18):3K(2)          R= +7644  p~=  3e-2302    FAIL !!!!!!!!
//|   [Low8/64]BRank(18):4K(8)          R=+23082  p~= 0           FAIL !!!!!!!!
//|   [Low8/64]BRank(18):6K(2)          R=+19335  p~=  2e-5821    FAIL !!!!!!!!
//|   [Low8/64]BRank(18):8K(4)          R=+38366  p~= 0           FAIL !!!!!!!!
//|   [Low8/64]BRank(18):12K(1)         R=+30205  p~=  1e-9093    FAIL !!!!!!!!
//|   [Low8/64]BRank(18):16K(4)         R=+82453  p~= 0           FAIL !!!!!!!!
//|   [Low8/64]BRank(18):24K(1)         R=+63271  p~= 0           FAIL !!!!!!!!
//|   [Low8/64]BRank(18):32K(2)         R=+120653 p~= 0           FAIL !!!!!!!!
//|   [Low8/64]BRank(18):48K(1)         R=+129402 p~= 0           FAIL !!!!!!!!
//|   [Low8/64]BRank(18):64K(1)         R=+174868 p~= 0           FAIL !!!!!!!!
//|   ...and 1698 test result(s) without anomalies
//| 
//| rng=RNG_stdin64, seed=0x88ab2def
//| length= 16 terabytes (2^44 bytes), time= 1067251 seconds
//|   Test Name                         Raw       Processed     Evaluation
//|   BRank(18):12K(16)                 R= +5089  p~=  4e-3746    FAIL !!!!!!!!
//|   BRank(18):16K(64)                 R=+21199  p~= 0           FAIL !!!!!!!!
//|   BRank(18):24K(16)                 R=+21622  p~= 0           FAIL !!!!!!!!
//|   BRank(18):32K(32)                 R=+46165  p~= 0           FAIL !!!!!!!!
//|   BRank(18):48K(8)                  R=+38670  p~= 0           FAIL !!!!!!!!
//|   BRank(18):64K(22)                 R=+89975  p~= 0           FAIL !!!!!!!!
//|   DC6-5x4Bytes-1                    R= +12.4  p =  9.8e-8   very suspicious
//|   [Low1/8]BRank(18):1536(8)         R= +3598  p~=  2e-2310    FAIL !!!!!!!!
//|   [Low1/8]BRank(18):2K(16)          R=+10600  p~=  3e-7802    FAIL !!!!!!!!
//|   [Low1/8]BRank(18):3K(4)           R=+10811  p~=  1e-5750    FAIL !!!!!!!!
//|   [Low1/8]BRank(18):4K(16)          R=+32644  p~= 0           FAIL !!!!!!!!
//|   [Low1/8]BRank(18):6K(4)           R=+27344  p~= 0           FAIL !!!!!!!!
//|   [Low1/8]BRank(18):8K(8)           R=+54257  p~= 0           FAIL !!!!!!!!
//|   [Low1/8]BRank(18):12K(2)          R=+42716  p~= 0           FAIL !!!!!!!!
//|   [Low1/8]BRank(18):16K(8)          R=+116607 p~= 0           FAIL !!!!!!!!
//|   [Low1/8]BRank(18):24K(2)          R=+89478  p~= 0           FAIL !!!!!!!!
//|   [Low1/8]BRank(18):32K(4)          R=+170629 p~= 0           FAIL !!!!!!!!
//|   [Low1/8]BRank(18):48K(1)          R=+129402 p~= 0           FAIL !!!!!!!!
//|   [Low1/8]BRank(18):64K(2)          R=+245352 p~= 0           FAIL !!!!!!!!
//|   [Low1/16]BRank(18):768(4)         R= +2544  p~=  4e-1354    FAIL !!!!!!!!
//|   [Low1/16]BRank(18):1K(16)         R=+10600  p~=  3e-7802    FAIL !!!!!!!!
//|   [Low1/16]BRank(18):1536(4)        R=+10811  p~=  1e-5750    FAIL !!!!!!!!
//|   [Low1/16]BRank(18):2K(8)          R=+23082  p~= 0           FAIL !!!!!!!!
//|   [Low1/16]BRank(18):3K(2)          R=+19335  p~=  2e-5821    FAIL !!!!!!!!
//|   [Low1/16]BRank(18):4K(8)          R=+54257  p~= 0           FAIL !!!!!!!!
//|   [Low1/16]BRank(18):6K(2)          R=+42716  p~= 0           FAIL !!!!!!!!
//|   [Low1/16]BRank(18):8K(4)          R=+82453  p~= 0           FAIL !!!!!!!!
//|   [Low1/16]BRank(18):12K(1)         R=+63271  p~= 0           FAIL !!!!!!!!
//|   [Low1/16]BRank(18):16K(4)         R=+170629 p~= 0           FAIL !!!!!!!!
//|   [Low1/16]BRank(18):24K(1)         R=+129402 p~= 0           FAIL !!!!!!!!
//|   [Low1/16]BRank(18):32K(2)         R=+245352 p~= 0           FAIL !!!!!!!!
//|   [Low1/16]BRank(18):48K(1)         R=+261665 p~= 0           FAIL !!!!!!!!
//|   [Low1/16]BRank(18):64K(1)         R=+349841 p~= 0           FAIL !!!!!!!!
//|   [Low1/32]BRank(18):384(4)         R= +2544  p~=  4e-1354    FAIL !!!!!!!!
//|   [Low1/32]BRank(18):512(16)        R=+10600  p~=  3e-7802    FAIL !!!!!!!!
//|   [Low1/32]BRank(18):768(4)         R=+10811  p~=  1e-5750    FAIL !!!!!!!!
//|   [Low1/32]BRank(18):1K(8)          R=+23082  p~= 0           FAIL !!!!!!!!
//|   [Low1/32]BRank(18):1536(2)        R=+19335  p~=  2e-5821    FAIL !!!!!!!!
//|   [Low1/32]BRank(18):2K(8)          R=+54257  p~= 0           FAIL !!!!!!!!
//|   [Low1/32]BRank(18):3K(2)          R=+42716  p~= 0           FAIL !!!!!!!!
//|   [Low1/32]BRank(18):4K(4)          R=+82453  p~= 0           FAIL !!!!!!!!
//|   [Low1/32]BRank(18):6K(1)          R=+63271  p~= 0           FAIL !!!!!!!!
//|   [Low1/32]BRank(18):8K(4)          R=+170629 p~= 0           FAIL !!!!!!!!
//|   [Low1/32]BRank(18):12K(1)         R=+129402 p~= 0           FAIL !!!!!!!!
//|   [Low1/32]BRank(18):16K(2)         R=+245352 p~= 0           FAIL !!!!!!!!
//|   [Low1/32]BRank(18):24K(1)         R=+261665 p~= 0           FAIL !!!!!!!!
//|   [Low1/32]BRank(18):32K(2)         R=+494750 p~= 0           FAIL !!!!!!!!
//|   [Low1/32]BRank(18):48K(1)         R=+526192 p~= 0           FAIL !!!!!!!!
//|   [Low1/64]BRank(18):256(16)        R=+10600  p~=  3e-7802    FAIL !!!!!!!!
//|   [Low1/64]BRank(18):384(4)         R=+10811  p~=  1e-5750    FAIL !!!!!!!!
//|   [Low1/64]BRank(18):512(8)         R=+23082  p~= 0           FAIL !!!!!!!!
//|   [Low1/64]BRank(18):768(4)         R=+27344  p~= 0           FAIL !!!!!!!!
//|   [Low1/64]BRank(18):1K(8)          R=+54257  p~= 0           FAIL !!!!!!!!
//|   [Low1/64]BRank(18):1536(2)        R=+42716  p~= 0           FAIL !!!!!!!!
//|   [Low1/64]BRank(18):2K(4)          R=+82453  p~= 0           FAIL !!!!!!!!
//|   [Low1/64]BRank(18):3K(2)          R=+89478  p~= 0           FAIL !!!!!!!!
//|   [Low1/64]BRank(18):4K(4)          R=+170629 p~= 0           FAIL !!!!!!!!
//|   [Low1/64]BRank(18):6K(1)          R=+129402 p~= 0           FAIL !!!!!!!!
//|   [Low1/64]BRank(18):8K(2)          R=+245352 p~= 0           FAIL !!!!!!!!
//|   [Low1/64]BRank(18):12K(1)         R=+261665 p~= 0           FAIL !!!!!!!!
//|   [Low1/64]BRank(18):16K(2)         R=+494750 p~= 0           FAIL !!!!!!!!
//|   [Low1/64]BRank(18):24K(1)         R=+526192 p~= 0           FAIL !!!!!!!!
//|   [Low1/64]BRank(18):32K(1)         R=+702543 p~= 0           FAIL !!!!!!!!
//|   [Low4/16]BRank(18):3K(8)          R= +3598  p~=  2e-2310    FAIL !!!!!!!!
//|   [Low4/16]BRank(18):4K(32)         R=+14990  p~= 0           FAIL !!!!!!!!
//|   [Low4/16]BRank(18):6K(8)          R=+15289  p~=  4e-9814    FAIL !!!!!!!!
//|   [Low4/16]BRank(18):8K(16)         R=+32644  p~= 0           FAIL !!!!!!!!
//|   [Low4/16]BRank(18):12K(4)         R=+27344  p~= 0           FAIL !!!!!!!!
//|   [Low4/16]BRank(18):16K(16)        R=+76731  p~= 0           FAIL !!!!!!!!
//|   [Low4/16]BRank(18):24K(4)         R=+60410  p~= 0           FAIL !!!!!!!!
//|   [Low4/16]BRank(18):32K(8)         R=+116607 p~= 0           FAIL !!!!!!!!
//|   [Low4/16]BRank(18):48K(2)         R=+89478  p~= 0           FAIL !!!!!!!!
//|   [Low4/16]BRank(18):64K(5)         R=+190769 p~= 0           FAIL !!!!!!!!
//|   [Low4/32]BRank(18):1536(8)        R= +3598  p~=  2e-2310    FAIL !!!!!!!!
//|   [Low4/32]BRank(18):2K(16)         R=+10600  p~=  3e-7802    FAIL !!!!!!!!
//|   [Low4/32]BRank(18):3K(4)          R=+10811  p~=  1e-5750    FAIL !!!!!!!!
//|   [Low4/32]BRank(18):4K(16)         R=+32644  p~= 0           FAIL !!!!!!!!
//|   [Low4/32]BRank(18):6K(4)          R=+27344  p~= 0           FAIL !!!!!!!!
//|   [Low4/32]BRank(18):8K(8)          R=+54257  p~= 0           FAIL !!!!!!!!
//|   [Low4/32]BRank(18):12K(2)         R=+42716  p~= 0           FAIL !!!!!!!!
//|   [Low4/32]BRank(18):16K(8)         R=+116607 p~= 0           FAIL !!!!!!!!
//|   [Low4/32]BRank(18):24K(2)         R=+89478  p~= 0           FAIL !!!!!!!!
//|   [Low4/32]BRank(18):32K(4)         R=+170629 p~= 0           FAIL !!!!!!!!
//|   [Low4/32]BRank(18):48K(1)         R=+129402 p~= 0           FAIL !!!!!!!!
//|   [Low4/32]BRank(18):64K(2)         R=+247300 p~= 0           FAIL !!!!!!!!
//|   [Low4/64]BRank(18):768(4)         R= +2544  p~=  4e-1354    FAIL !!!!!!!!
//|   [Low4/64]BRank(18):1K(16)         R=+10600  p~=  3e-7802    FAIL !!!!!!!!
//|   [Low4/64]BRank(18):1536(4)        R=+10811  p~=  1e-5750    FAIL !!!!!!!!
//|   [Low4/64]BRank(18):2K(8)          R=+23082  p~= 0           FAIL !!!!!!!!
//|   [Low4/64]BRank(18):3K(2)          R=+19335  p~=  2e-5821    FAIL !!!!!!!!
//|   [Low4/64]BRank(18):4K(8)          R=+54257  p~= 0           FAIL !!!!!!!!
//|   [Low4/64]BRank(18):6K(2)          R=+42716  p~= 0           FAIL !!!!!!!!
//|   [Low4/64]BRank(18):8K(4)          R=+82453  p~= 0           FAIL !!!!!!!!
//|   [Low4/64]BRank(18):12K(1)         R=+63271  p~= 0           FAIL !!!!!!!!
//|   [Low4/64]BRank(18):16K(4)         R=+170629 p~= 0           FAIL !!!!!!!!
//|   [Low4/64]BRank(18):24K(1)         R=+129402 p~= 0           FAIL !!!!!!!!
//|   [Low4/64]BRank(18):32K(2)         R=+247300 p~= 0           FAIL !!!!!!!!
//|   [Low4/64]BRank(18):48K(1)         R=+351219 p~= 0           FAIL !!!!!!!!
//|   [Low4/64]BRank(18):64K(1)         R=+527569 p~= 0           FAIL !!!!!!!!
//|   [Low8/32]BRank(18):3K(8)          R= +3598  p~=  2e-2310    FAIL !!!!!!!!
//|   [Low8/32]BRank(18):4K(32)         R=+14990  p~= 0           FAIL !!!!!!!!
//|   [Low8/32]BRank(18):6K(8)          R=+15289  p~=  4e-9814    FAIL !!!!!!!!
//|   [Low8/32]BRank(18):8K(16)         R=+32644  p~= 0           FAIL !!!!!!!!
//|   [Low8/32]BRank(18):12K(4)         R=+27344  p~= 0           FAIL !!!!!!!!
//|   [Low8/32]BRank(18):16K(16)        R=+76731  p~= 0           FAIL !!!!!!!!
//|   [Low8/32]BRank(18):24K(4)         R=+60410  p~= 0           FAIL !!!!!!!!
//|   [Low8/32]BRank(18):32K(8)         R=+116607 p~= 0           FAIL !!!!!!!!
//|   [Low8/32]BRank(18):48K(2)         R=+89478  p~= 0           FAIL !!!!!!!!
//|   [Low8/32]BRank(18):64K(5)         R=+190769 p~= 0           FAIL !!!!!!!!
//|   [Low8/64]BRank(18):1536(8)        R= +3598  p~=  2e-2310    FAIL !!!!!!!!
//|   [Low8/64]BRank(18):2K(16)         R=+10600  p~=  3e-7802    FAIL !!!!!!!!
//|   [Low8/64]BRank(18):3K(4)          R=+10811  p~=  1e-5750    FAIL !!!!!!!!
//|   [Low8/64]BRank(18):4K(16)         R=+32644  p~= 0           FAIL !!!!!!!!
//|   [Low8/64]BRank(18):6K(4)          R=+27344  p~= 0           FAIL !!!!!!!!
//|   [Low8/64]BRank(18):8K(8)          R=+54257  p~= 0           FAIL !!!!!!!!
//|   [Low8/64]BRank(18):12K(2)         R=+42716  p~= 0           FAIL !!!!!!!!
//|   [Low8/64]BRank(18):16K(8)         R=+116607 p~= 0           FAIL !!!!!!!!
//|   [Low8/64]BRank(18):24K(2)         R=+89478  p~= 0           FAIL !!!!!!!!
//|   [Low8/64]BRank(18):32K(4)         R=+170629 p~= 0           FAIL !!!!!!!!
//|   [Low8/64]BRank(18):48K(1)         R=+129402 p~= 0           FAIL !!!!!!!!
//|   [Low8/64]BRank(18):64K(2)         R=+247300 p~= 0           FAIL !!!!!!!!
//|   ...and 1738 test result(s) without anomalies
//
// (it is, perhaps, worth noting that XorShift* 64/32, a generator half the
// size of Xoroshiro128+ does not suffer these kinds of failures)

// - 128 state bits, uint64_t output, period 2^128 - 1
using xoroshiro128plus64v0_1 =
    xoroshiro_detail::xoroshiro_plus<uint64_t, uint64_t, 55, 14, 36>;

using xoroshiro128plus64v1_0 =
    xoroshiro_detail::xoroshiro_plus<uint64_t, uint64_t, 24, 16, 37>;

using xoroshiro128plus64 = xoroshiro128plus64v1_0;

using xoroshiro128starstar64v1_0 =
    xoroshiro_detail::xoroshiro_starstar<uint64_t, uint64_t, 24, 16, 37,
                                         5, 7, 9>;

using xoroshiro128starstar64 = xoroshiro128starstar64v1_0;

// This is a variation on the above generator which tries to avoid the
// problems with weak low-order bits by discarding the low-order bits,
// to return only the high 32 bits.

// - 128 state bits, uint32_t output, period 2^128 - 1
using xoroshiro128plus32v0_1 =
    xoroshiro_detail::xoroshiro_plus<uint64_t, uint32_t, 55, 14, 36>;

using xoroshiro128plus32v1_0 =
    xoroshiro_detail::xoroshiro_plus<uint64_t, uint32_t, 24, 16, 37>;

using xoroshiro128plus32 = xoroshiro128plus32v1_0;
    
//// Variations ////

// These are variations on the above theme. The constants are
// mathematically correct and are designed to have a similar size
// relationships to the constants for Xoroshiro128+.  Empirical
// testing shows them to be acceptable, but it is possible that there
// are even better constants.

// These variations have similar properties to the ones above.  The
// first variant fails statistical tests due to weak low-order bits,
// and the second probably passes stringent statistical tests.

// 256 //

#if __SIZEOF_INT128__
// - 256 state bits, __uint128_t output, period 2^256 - 1
using xoroshiro256plus128 =
    xoroshiro_detail::xoroshiro_plus<__uint128_t, __uint128_t, 105, 36, 70>;

// - 256 state bits, uint64_t output, period 2^256 - 1
using xoroshiro256plus64 =
    xoroshiro_detail::xoroshiro_plus<__uint128_t, uint64_t, 105, 36, 70>;
#endif

// 64 //

// - 64 state bits, uint32_t output, period 2^64 - 1
using xoroshiro64plus32vMEO1 =
    xoroshiro_detail::xoroshiro_plus<uint32_t, uint32_t, 27, 7, 20>;

using xoroshiro64plus32v1_0 =
    xoroshiro_detail::xoroshiro_plus<uint32_t, uint32_t, 26, 9, 13>;

using xoroshiro64plus32 = xoroshiro64plus32v1_0;

using xoroshiro64star32v1_0 =
    xoroshiro_detail::xoroshiro_star<uint32_t, uint32_t, 26, 9, 13, 0x9E3779BB>;

using xoroshiro64star32 = xoroshiro64star32v1_0;

using xoroshiro64starstar32v1_0 =
    xoroshiro_detail::xoroshiro_starstar<uint32_t, uint32_t, 26, 9, 13,
                                         0x9E3779BB, 5, 5>;

using xoroshiro64starstar32 = xoroshiro64starstar32v1_0;


// - 64 state bits, uint16_t output, period 2^64 - 1
using xoroshiro64plus16vMEO1 =
    xoroshiro_detail::xoroshiro_plus<uint32_t, uint16_t, 27, 7, 20>;

using xoroshiro64plus16v1_0 =
    xoroshiro_detail::xoroshiro_plus<uint32_t, uint16_t, 26, 9, 13>;

using xoroshiro64plus16 = xoroshiro64plus16v1_0;

// 32 //

// - 32 state bits, uint16_t output, period 2^32 - 1
using xoroshiro32plus16 =
    xoroshiro_detail::xoroshiro_plus<uint16_t, uint16_t, 13, 5, 10>;

using xoroshiro32star16 =
    xoroshiro_detail::xoroshiro_star<uint16_t, uint16_t, 13, 5, 10, 0x79BB>;

// - 32 state bits, uint8_t output, period 2^32 - 1
using xoroshiro32plus8 =
    xoroshiro_detail::xoroshiro_plus<uint16_t, uint8_t, 13, 5, 10>;

// 16 //

// - 16 state bits, uint8_t output, period 2^16 - 1
using xoroshiro16plus8 =
    xoroshiro_detail::xoroshiro_plus<uint8_t, uint8_t, 4, 7, 3>;

using xoroshiro16star8 =
    xoroshiro_detail::xoroshiro_star<uint8_t, uint8_t, 4, 7, 3, 0xBB>;

#endif // XOROSHIRO_HPP_INCLUDED
