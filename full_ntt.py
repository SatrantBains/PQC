class full_ntt():
    def __init__(self, size):
        self.memory = [1]*size

    def orderReverse(self, poly, N_bit):
        """docstring for order"""
        for i, coeff in enumerate(poly):
            rev_i = self.bitReverse(i, N_bit)
            if rev_i > i:
                coeff ^= poly[rev_i]
                poly[rev_i] ^= coeff
                coeff ^= poly[rev_i]
                poly[i] = coeff
        return poly
        # Bit Reverse 

    def bitReverse(self, num, len):
        """
        integer bit reverse
        input: num, bit length
        output: rev_num 
        example: input 6(11 0) output3(011)
        complexity: O(len)
        """
        rev_num = 0
        for i in range(0, len):
            test = num >> i
            test2 = test & 1
            if (num >> i) & 1:
                rev_num |= 1 << (len - 1 - i)
        return rev_num
    
    def modInv(self, x, M):
        t, new_t, r, new_r = 0, 1, M, x

        while new_r != 0:
            quotient = int(r / new_r)
            tmp_new_t = new_t
            tmp_t = t
            tmp_new_r = new_r
            tmp_r = r
            t, new_t = new_t, (t - quotient * new_t)
            r, new_r = new_r, (r % new_r)
        if r > 1:
            return "x is not invertible."
        if t < 0:
            t = t + M
        return t
    
    def idx_access_gen(self,n):
        h = 2
        idx_etemp = []
        idx_otemp = []
        idx_even = []
        idx_odd = []

        while h <= n:
            hf = h // 2
            for i in range(0, n, h):
                for j in range(hf):
                    idx_etemp.append(i+j), idx_otemp.append(i+j+hf)
            idx_even.append(idx_etemp), idx_odd.append(idx_otemp)
            idx_etemp = []
            idx_otemp = []
            h *= 2
        return idx_even, idx_odd
    
    def addr_gen(self, n, layer, log2n, num_bf=4, N_mem=128):
        addr_rEven = []
        addr_rOdd = []
        gap = 2**layer
        idx_Even, idx_Odd = self.idx_access_gen(8) 
        n_div = N_mem // 2
        if layer == 0:
            for i in range(num_bf):
                addr_rEven.append(i) 
                addr_rOdd.append(i)
        else:
            for i in range(log2n-layer):
                addr_EvenBase = idx_Even[layer][i] + i*gap*2
                addr_OddBase = idx_Odd[layer][i] + i*gap*2
                # idx_EvenNew = idx_Even[layer][i] % 
                for j in range(gap):
                    addr_rEven.append( (addr_EvenBase + j) // n_div)
                    addr_rOdd.append( (addr_OddBase + j) // n_div)  
        return addr_rEven, addr_rOdd

    def addr_gen2(self, n, count, num_bf=8, N_mem=128):
        addr_rEven = []
        addr_rOdd = []
        idx_even = []
        idx_odd = []

        c = n//2//num_bf
        n_coef = N_mem // 8
        layer = count // c
        step = count % c
        diff = 2 ** layer 

        # idx_rd0 = 2*((step*num_bf) // diff)*diff+((step*num_bf) % diff)
        # idx_rd1 = idx_rd0+diff

        idx_rdEven, idx_rdOdd = self.idx_access_gen(8) 
        idx_rd0 = idx_rdEven[count][0]
        idx_rd1 = idx_rdOdd[count][0]

        parts = num_bf // diff if num_bf >= diff else 1
        print("parts =", parts)
        for i in range(parts):

            addr_rd0_base = idx_rd0 + i*diff*2
            addr_rd1_base = idx_rd1 + i*diff*2

            for j in range(diff):
                addr_rEven.append( (addr_rd0_base + j) // n_coef)
                addr_rOdd.append( (addr_rd1_base + j) // n_coef)  
                idx_even.append( (addr_rd0_base + j) % n_coef )
                idx_odd.append( (addr_rd1_base + j) % n_coef )

        return addr_rEven, addr_rOdd, idx_even, idx_odd

    def create_a(self, n):
        ans = []
        for i in range(n):
            ans.append(i)
        return ans

    def dual_port_bram_unit(self, readaddr, writeaddr, datain, reade, writee):

        # initializing 5 memory location with one polynomials like [1,2,3,4,5,6,7,8] for testing
        # in this case each polynomial has 8 coefficients is 8 bits and takes 64 bits total 
    
        if reade == 1:
            return self.memory[readaddr]
        if writee == 1:
            self.memory[writeaddr] = datain
        elif(reade == 1 and writee == 1):
            self.memory[writeaddr] = datain
            return self.memory[readaddr]
    

        
    def tf_gen_lookup(self, n, w, q):

        w_array = [1]*(n//2)
        for i in range(1, n // 2):
            w_array[i] = (w**i)%q # generating roots that we will need

        return w_array


    def ct_butterfly(self, even, odd, w, q):
        even = self.bintodec(even,8)[0]
        odd = self.bintodec(odd,8)[0]
        odd = odd*w
        a = (even + odd)%q
        b = (even - odd)%q

        # u = self.montgomery_reduction(a,32,q)
        # v = self.montgomery_reduction(b,32,q)

        return a,b
 
    def montgomery_reduction(self, T, R, m):
        # Step 1: Convert T to Montgomery form
        Ttemp = T * R
        T_prime = Ttemp % m  # conversion of T to Montgomery Form
        
        # Step 2: Use Fermat's Little Theorem to find R^-1
        FLT = pow(R, m - 2, m)  # R^(m-2) mod m gives R^-1
        R_inv = FLT % m  # Ensures R_inv is reduced mod m
        
        # Step 3: Perform the reduction
        ptemp = T_prime * R_inv
        p = ptemp % m  # reduced value of p

        # Step 4: Calculate intermediate result
        result = T_prime + (m * p)

        # Step 5: Ensure result is less than m
        while result >= m:
            result -= m

        return result  # Return the final result
    
    def cntrl_unit(self, layer):
        #we now have idx_access_gen which generates the indeces we need to pull for the ntt
        # idx_even = ( [0,2,4,6], [0,1,4,5], [0,1,2,3] )
        # idx_odd = ( [1,3,5,7], [2,3,6,7], [4,5,6,7] )
        idx_even, idx_odd = self.idx_access_gen(self.n)

        addr = [[0],[5],[5]]
        return idx_even[layer], idx_odd[layer]
    
    def bintodec(self, a, n):
        temp = [1]*( int(len(a)/n) ) 
        poly = [1]*( int(len(a)/n) ) 

        for i in range( int(len(a)/n) ):
            # grab n bit windows
            temp = a[i*n:i*n+n]
            #save to poly array
            poly[i] = int(temp,2)
        return poly 
    

    def dectobin(self, a, n):
        test = ''
        if type(a) == int:
            temp = bin(a)[2:]
            while( len(temp) < n):
                temp = '0' + temp
            test += temp
            return test
        else:
            for i in range(len(a)):
                temp = bin(a[i])[2:]

                while( len(temp) < n):
                    temp = '0' + temp
                test += temp
            return test
            
    # going to change significantly as we need to generate the mem access pattern with readaddr and writeadrr
    # control unit will contain the correct indeces needed
    def full_scale_ntt(self, addr, q, w, n, bits):
        # q = 17
        # w = 9
        # n = 8
        # bf_cnt = 4
        log2n = 3
        
        # Pre-processing polynomial bram -> bin -> decimal -> reversal -> stored back to bram 

        # addr = 0, datain = 0, reade = 1, writee = 0, initialization = 1
        a = self.one_port_bram_unit(addr,0,1,0)

        poly = self.bintodec(a,n)

        # print("DECODED POLY: ", poly)

        # order reversing our polynomial 
        N_bit = n.bit_length() - 1
        rev_poly = self.orderReverse(poly, N_bit)
        poly = rev_poly 
        
        # print(poly)
        datain = self.dectobin(poly,n)

        # reversed polynomial sent back to memory 
        self.one_port_bram_unit(addr, datain, 0, 1)


        # test to see if reversal worked
        # b = self.one_port_bram_unit(i,0,1,0,0)
        # print(b)
        # b = self.bintodec(b,n)

        w_array = self.tf_gen_lookup(n,w,q)
        # print("ROOTS ARE: ", w_array)
        pattern = []
        idx_evenf, idx_oddf = self.idx_access_gen(n)

        for i in range( log2n ):
            # Pulling correct indeces from cntrl_unit
            idx_even,idx_odd = idx_evenf[i], idx_oddf[i]

            # Pulling data from bram location 0
            bin_poly = self.one_port_bram_unit(addr, 0, 1, 0)

            print("i(layer):", i)
            print("poly:", self.bintodec(bin_poly,8), "\n")
            
            for j in range(n//2):
                twiddle_idx = (j * n) // (2**(i+1))
                twiddle_factor = w_array[twiddle_idx % (n//2)] 
                
                # Using our idx_even/odd indeces to splice the data that we need    
                # For example if we need index 0 we grab bits 1-8, if we need index 1 we grab 2-10...
                even = bin_poly[ idx_even[j]*bits:idx_even[j]*bits+bits ]
                odd = bin_poly[ idx_odd[j]*bits:idx_odd[j]*bits+bits ]

                # print("Even: ", even)
                # print("Odd: ", odd, " TWIDDLE FACTOR IS: " , twiddle_factor)
                # print("odd = %d, even = %d " % (idx_odd[j],  idx_even[j]))

                # Data sent as binary converted interally to integers for calculation
                u,v = self.ct_butterfly(even,odd,twiddle_factor,q)
            
                u = self.dectobin(u,bits)
                v = self.dectobin(v,bits)

                # Python strings are immutable so to get around that I am simply splicing and inserting the changed data back into the string 
                bin_poly = bin_poly[:idx_even[j]*bits] + u + bin_poly[idx_even[j]*bits+bits:]
                bin_poly = bin_poly[:idx_odd[j]*bits] + v + bin_poly[idx_odd[j]*bits+bits:]
                pattern.append((idx_even[j], idx_odd[j], twiddle_idx % (n//2), twiddle_factor))
                # Store binary data back into bram
                self.one_port_bram_unit(addr, bin_poly, 0, 1)

        poly = self.one_port_bram_unit(addr,0,1,0)
        poly = self.bintodec(poly,8)
        return poly, pattern
    
    def small_scale_intt(self, addr, q, n, w, bits):
        log2n = 3
        a = self.one_port_bram_unit(addr,0,1,0)

        poly = self.bintodec(a,n)

        # print("DECODED POLY: ", poly)

        # order reversing our polynomial 
        N_bit = n.bit_length() - 1
        # rev_poly = self.orderReverse(poly, N_bit)
        # poly = rev_poly 
        
        # print(poly)
        datain = self.dectobin(poly,n)

        # reversed polynomial sent back to memory 
        self.one_port_bram_unit(addr, datain, 0, 1)

        w_array = self.tf_gen_lookup(n,w,q)

        inverted_w = self.modInv(w,q)
        inverted_n = self.modInv(n,q)

        # print(inverted_w)
        # print(inverted_n)
        pattern = []
        poly,pattern = self.full_scale_ntt(addr,q,inverted_w,n,0,bits)

        for i in range(0, n):
            poly[i] = (poly[i] * inverted_n) % q

        return poly,pattern

        


from sympy import ntt 

if __name__ == '__main__':

    ntt = full_ntt(32) # -> we need 16 memory locations for n = 256 where each bram addr contains 16 coefficients 
    even,odd = (ntt.idx_access_gen(8))
    print(even)
    print(odd)
    # poly = ntt.create_a(256)
    # test,test2= ntt.addr_gen(8,2,3,4,16)
    # print(test)
    # print(test2)
    for i in range (3):
        test3,test4,test5,test6 = ntt.addr_gen2(8,i,4,16)
        print(test3)
        print(test4)
        print(test5)
        print(test6)
    # test2 = ntt.dectobin(mem5,16)
    # print(test2)
    # test3 = ntt.bintodec(test2,16)
    # print(test3)
    
    #Generating 32 memory locations with 8 16 bit coeffiecients each 
    # for i in range(32):
    #     mem = poly[i*8:i*8+8]
    #     memtemp = ntt.dectobin(mem,16)
    #     ntt.dual_port_bram_unit(i,memtemp,0,1)
    # for i in range(7):
    #     test,test2 = ntt.addr_gen(256,i,4,128)
    #     print("EVEN: ", test)
    #     print("ODD ", test2)
        
    # mem5 = poly[:128]
    # mem6 = poly[128:]

    # mem5 = ntt.dectobin(mem5, 16)
    # mem6 = ntt.dectobin(mem6, 16)

    # ntt.one_port_bram_unit(5,mem5,0,1,1)
    # ntt.one_port_bram_unit(6,mem6,0,1,0)
    # b = ntt.one_port_bram_unit(5,0,1,0,0)
    # print(b)
    # idx_even, idx_odd = ntt.idx_access_gen(256)
    # print(idx_even, "\n")
    # print(idx_odd)

    # q = 17
    # w = 9
    # n = 8

    # a = [1,2,3,4,5,6,7,8]
 
    # ntt2 = ntt(a,17)
    
    # TO DO ->  Split 256 polynomial into two different memory locations -> Figure out how to generalize
    # print("###################### NTT ######################")
    # print("Starting Polynomial: ", a)
    # ntt = full_ntt(10)

    # ntt1,pattern = ntt.full_scale_ntt(0, q, w, n, 1,8)
    # print("Transformed Polynomial: ", ntt1)
    # print("Confirmed Result : ", ntt2 , "\n")
    # print("Access Pattern: ", pattern , "\n")

    # print("###################### INTT ######################")
    # print("Starting Polynomial: ", ntt1)
    # intt, patti = ntt.small_scale_intt(0,q,n,w,8)
    # print("Transformed Polynomial ", intt, "\n")
    # print("Access Pattern: ", patti , "\n")
