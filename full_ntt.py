
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

        idx_tftemp = []
        idx_tf = []

        while h <= n:
            hf = h // 2
            ut = n // h

            for i in range(0, n, h):
                for j in range(hf):
                    idx_etemp.append(i+j), idx_otemp.append(i+j+hf), idx_tftemp.append(ut * j)
            idx_even.append(idx_etemp), idx_odd.append(idx_otemp), idx_tf.append(idx_tftemp)
            idx_etemp = []
            idx_otemp = []
            idx_tftemp = []
            h *= 2
        
        return idx_even, idx_odd, idx_tf
    
    def bf_cnt_splice(self, arr, bf_cnt):
        return [[row[i:i+4] for i in range(0, len(row), bf_cnt)] for row in arr]   

    def addr_gen(self, n, count,layer, idxe, idxo, num_bf=8, N_mem=128, N_bits = 8):
        addr_rEven = []
        addr_rOdd = []
        idx_even = []
        idx_odd = []

        c = n//2//num_bf
        n_coef = N_mem // N_bits # 16 = bits per coef
        # layer = count // c
        diff = 2 ** layer


        idx_rd0 = idxe[count][0]
        idx_rd1 = idxo[count][0]

        parts = num_bf // diff if num_bf >= diff else 1
        # print("parts =", parts)
        for i in range(parts):

            addr_rd0_base = idx_rd0 + i*diff*2
            addr_rd1_base = idx_rd1 + i*diff*2

            for j in range(diff):
                addr_rEven.append( (addr_rd0_base + j) // n_coef)
                addr_rOdd.append( (addr_rd1_base + j) // n_coef)  
                idx_even.append( (addr_rd0_base + j) % n_coef )
                idx_odd.append( (addr_rd1_base + j) % n_coef )
    
        return addr_rEven, addr_rOdd, idx_even, idx_odd
    
    def addr_gen2(self, n, count, num_bf=8, N_mem=256, bits = 16):
        addr_rd0 = []
        addr_rd1 = []
        idx0 = []
        idx1 = []

        c = n//2//num_bf
        n_coef = N_mem // bits
        layer = count // c
        step = count % c
        diff = 2 ** layer 
        idx_rd0 = 2*((step*num_bf) // diff)*diff+((step*num_bf) % diff)
        idx_rd1 = idx_rd0+diff
        ans = []
        parts = num_bf // diff if num_bf >= diff else 1
        # print("parts =", parts)
        for i in range(parts):
            addr_rd0_base = idx_rd0 + i*diff*2
            addr_rd1_base = idx_rd1 + i*diff*2
            for j in range(diff):
                addr_rd0.append( (addr_rd0_base + j) // n_coef )
                addr_rd1.append( (addr_rd1_base + j) // n_coef )
                idx0.append( (addr_rd0_base + j) % n_coef )
                idx1.append( (addr_rd1_base + j) % n_coef )

                
        return addr_rd0 , addr_rd1, idx0, idx1
    
    def addr_gen_fin(self, n, log2n, idxe, idxo, bf_cnt, N_mem, N_bits):
        N_coef = N_mem//N_bits
        rows, cols = 8, 128  # specify dimensions
        idx_even = [[0] * cols for _ in range(rows)]
        idx_odd = [[0] * cols for _ in range(rows)] 
        addr_even = [[0] * cols for _ in range(rows)] 
        addr_odd = [[0] * cols for _ in range(rows)]      
          
        for i in range(log2n):
            for j in range(n//2):
                idx_even[i][j] = idxe[i][j] % N_coef
                idx_odd[i][j] = idxo[i][j] % N_coef
                addr_even[i][j] = idxe[i][j]//16
                addr_odd[i][j] = idxo[i][j]//16

        idx_Even = self.bf_cnt_splice(idx_even,bf_cnt)
        idx_Odd = self.bf_cnt_splice(idx_odd,bf_cnt)
        addr_Even = self.bf_cnt_splice(addr_even, bf_cnt)
        addr_Odd = self.bf_cnt_splice(addr_odd, bf_cnt)
        return addr_Even, addr_Odd, idx_Even, idx_Odd

    
    def pre_gen_addr(self, n, log2n, idxe, idxo, bf_cnt, N_mem, N_bits):
        addr_even_temp = []
        addr_odd_temp = []

        idx_even_temp = []
        idx_odd_temp = []
        
        addr_even = []
        addr_odd = []
        
        idx_Even = []
        idx_Odd = []
        
        idx_even = self.bf_cnt_splice(idxe,bf_cnt)
        idx_odd = self.bf_cnt_splice(idxo,bf_cnt)
        for i in range(log2n):
            for j in range( (n//2//bf_cnt)):
                adde, addo, idxe, idxo = self.addr_gen(n,j,i,idx_even[i],idx_odd[i], bf_cnt,N_mem, N_bits)
                addr_even_temp.append(adde)
                addr_odd_temp.append(addo)
                idx_even_temp.append(idxe)
                idx_odd_temp.append(idxo)
            addr_even.append(addr_even_temp), addr_odd.append(addr_odd_temp), idx_Even.append(idx_even_temp), idx_Odd.append(idx_odd_temp)
            addr_even_temp = []
            addr_odd_temp = []
            idx_even_temp = []
            idx_odd_temp = []
        return addr_even, addr_odd, idx_Even, idx_Odd
        

    def create_a(self, n):
        ans = []
        for i in range(n):
            ans.append(i+1)
        return ans
    
    def dual_port_bram_unit(self, addra, addrb, dataina, datainb, readea, readeb, writeea, writeeb):
        dataouta, dataoutb = None, None  
        
        # Port A
        if writeea:  
            self.memory[addra] = dataina
        if readea:  
            dataouta = self.memory[addra]
        
        # Port B
        if writeeb:  
            self.memory[addrb] = datainb
        if readeb:  
            dataoutb = self.memory[addrb]
        
        return dataouta, dataoutb
    
    def load_memory(self, n, n_addr, n_loc_size, n_bit_coeff ):
        offset = n_loc_size//n_bit_coeff
        poly = self.create_a(n)
        # print(poly)
        N_bit = n.bit_length() - 1
        poly = self.orderReverse(poly, N_bit)
        for i in range(n_addr):
            data = poly[i*offset:i*offset+offset]
            datatemp = self.dectobin(data,n_bit_coeff)
            self.dual_port_bram_unit(i,0,datatemp,0,0,0,1,0)

        
    def tf_gen_lookup(self, n, w, q, log2n, bf_cnt):

        w_array = [1]*(n//2)
        for i in range(1, n // 2):
            w_array[i] = (w**i)%q # generating roots that we will need

    
        return w_array


    def ct_butterfly(self, even, odd, w, q, n_bits):
        even = self.bintodec(even,n_bits)[0]
        odd = self.bintodec(odd,n_bits)[0]
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
    
    def cntrl_unit(self, layer, n):
        #we now have idx_access_gen which generates the indeces we need to pull for the ntt
        # idx_even = ( [0,2,4,6], [0,1,4,5], [0,1,2,3] )
        # idx_odd = ( [1,3,5,7], [2,3,6,7], [4,5,6,7] )
        idx_even, idx_odd = self.idx_access_gen(n)

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

    def bitwise_insert(self, data_out, idx, value, bits):
        # Convert data_out to a list for easier bit-level manipulation
        data_list = list(data_out)
        # Insert value at the specified index by replacing bits
        data_list[idx*bits:idx*bits + bits] = list(value)  # Replace bits at idx with value
        return ''.join(data_list)

                               
    # going to change significantly as we need to generate the mem access pattern with readaddr and writeadrr
    # control unit will contain the correct indeces needed
    def full_scale_ntt(self, q, w, n, log2n, bf_cnt, bits, N_mem):

        idxe, idxo, idx_tf = self.idx_access_gen(n)
        # idxe2, idxo2, idxtf2 = self.idx_access_gen(256)
        test = 128//16
        idx_tf = self.bf_cnt_splice(idx_tf, bf_cnt)

        w_array = self.tf_gen_lookup(n,w,q,log2n,bf_cnt)
        # print("ROOTS ARE: ", w_array)
        pattern = []

        # I am only generating one layer of index values -> need to fix 
        # should be dimensions 8x32x4
        # addr_evenr, addr_oddr, idx_evenr, idx_oddr = self.pre_gen_addr(n, log2n, idxe, idxo, bf_cnt, N_mem, bits)
        addr_evenr,addr_oddr, idx_evenr,idx_oddr = self.addr_gen_fin(n,log2n,idxe,idxo,bf_cnt,N_mem,bits)
        # n = 8 test
        # addr_even = [[0,1,2,3],[0,0,2,2],[0,0,1,1]]
        # addr_odd = [[0,1,2,3],[1,1,3,3],[2,2,3,3]]
        # idx_even = [[0,0,0,0],[0,1,0,1],[0,1,0,1]]
        # idx_odd = [[1,1,1,1],[0,1,0,1],[0,1,0,1]]
        #loading spliced idxs and addrs 

        # addr_event, addr_oddt, idx_event, idx_oddt = self.pre_gen_addr(256, 8, bf_cnt,128,16)
        even = [1]*bf_cnt
        odd = [1]*bf_cnt
        u = [1]*bf_cnt
        v = [1]*bf_cnt
        twiddle_inner_idx = [1]*bf_cnt

        for i in range( log2n ):
            twiddle_layer = idx_tf[i]
            addr_even = addr_evenr[i]
            addr_odd = addr_oddr[i]
            idx_even = idx_evenr[i]
            idx_odd = idx_oddr[i]
            # Pulling correct indeces from cntrl_unit
            # idx_even,idx_odd = idx_evenf[i], idx_oddf[i]
            # addr_even, addr_odd, idx_even, idx_odd = self.addr_gen(n,i,bf_cnt,16)

            #addra, addrb, dataina, datainb, readea, readeb, writea, writeb
            # bin_poly_even, bin_poly_odd = self.dual_port_bram_unit(addr_even,addr_odd,0,0,1,1,0,0)

            print("i(layer):#####################", i)
            # print("poly:", self.bintodec(bin_poly,8), "\n")

            for j in range (0, n//2//bf_cnt):
                # print("j(layer):", j)
                # print(addr_even,"\n",addr_odd,"\n",idx_even,"\n",idx_odd)

                # data_out_even,data_out_odd = self.dual_port_bram_unit(addr_even[i][j],addr_odd[i][j],0,0,1,1,0,0)

                for k in range(bf_cnt):
                    # print("k(layer):", k)
                    twiddle_inner_idx[k] = twiddle_layer[j][k]
                    data_out_even,data_out_odd = self.dual_port_bram_unit(addr_even[j][k],addr_odd[j][k],0,0,1,1,0,0)

                    even[k] = data_out_even[ idx_even[j][k]*bits:idx_even[j][k]*bits+bits ]
                    odd[k] = data_out_odd[ idx_odd[j][k]*bits:idx_odd[j][k]*bits+bits ]
                    
                    teste = self.bintodec(even[k],bits)
                    testo = self.bintodec(odd[k],bits)
                    # print("Even: ", even)
                    # print("Odd: ", odd, " TWIDDLE FACTOR IS: " , twiddle_factor)
                    # print("odd = %d, even = %d " % (idx_odd[j],  idx_even[j]))

                    # Data sent as binary converted interally to integers for calculation\
                    tf = w_array[twiddle_inner_idx[k]]
                    u[k],v[k] = self.ct_butterfly(even[k],odd[k],tf,q,bits)
            
                    u[k] = self.dectobin(u[k],bits)
                    v[k] = self.dectobin(v[k],bits)

                    # Python strings are immutable so to get around that I am simply splicing and inserting the changed data back into the string
                    if(addr_even[j][k] == addr_odd[j][k]):
                        data_in = self.bitwise_insert(data_out_even,idx_even[j][k],u[k], bits)
                        data_in = self.bitwise_insert(data_in, idx_odd[j][k], v[k], bits)  # Keeps in-between bits intact
                        self.dual_port_bram_unit(addr_even[j][k],0,data_in,0,0,0,1,0)
                    else:
                        data_in_even = data_out_even[:idx_even[j][k]*bits] + u[k] + data_out_even[idx_even[j][k]*bits+bits:]
                        data_in_odd = data_out_odd[:idx_odd[j][k]*bits] + v[k] + data_out_odd[idx_odd[j][k]*bits+bits:]
                        # pattern.append((idx_even[j][k], idx_odd[j][k]))
                            # Store binary data back into bram
                        self.dual_port_bram_unit(addr_even[j][k],addr_odd[j][k],data_in_even,data_in_odd,0,0,1,1)

        # poly = self.bintodec(poly,8)
        poly = ''
        for i in range(N_mem//bits):
            poly += self.dual_port_bram_unit(i,0,0,0,1,0,0,0)[0] 
        poly = self.bintodec(poly,bits)
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
    n_mem_loc = 32 


    ntte = full_ntt(n_mem_loc) 

    # seq = ntte.create_a(256)
    # test = ntt(seq,18433)
    # print(test)
    # test2 = ntt(test, 18433, True)
    # print(test2)

    # # N = 8 
    # q = 17
    # n = 8
    # w = 9
    # num_bf = 4
    # N_mem = 4
    # bits = 16
    # n_bits_coef = 8

    # # N = 16 
    # q = 97
    # n = 16
    # w = 8
    # num_bf = 4
    # N_mem = 4
    # n_bits_coef = 16

    # N = 256 -
    q = 18433
    n = 256
    w = 5329
    num_bf = 4
    N_mem = 256
    n_bits_coef = 16



    ntte.load_memory(256,16,256,16)
    # ntte.load_memory(8,4,16,8)


    # test,test2 = ntte.full_scale_ntt(q,w,n,3,4,8,16)
    # print(test)
 
    final,test= ntte.full_scale_ntt(q,w,n,8,4,n_bits_coef,N_mem)
    print(final)



