library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.NUMERIC_STD.ALL;


entity cntrl_unit is
	generic (
				N : integer := 256;
				log2n : integer := 8;
				depth : integer := 7
				);
    Port ( clk : in STD_LOGIC;
           reset : in STD_LOGIC;
           addr_1 : out STD_LOGIC_VECTOR(depth-1 downto 0);
           done : out STD_LOGIC := '0';
           layer : out integer range 0 to log2n-1;
           step_o : out integer range 0 to N/2;
			  count1 : out integer;
			  count2 : out integer
           );
end cntrl_unit;


--  rename to read and write address instead of addr1 and addr2

architecture behavior of cntrl_unit is
    type state_ntt is (reset_state, compute_state, done_state);
   
    constant count_total : integer := N/2; -- only holds for one bf\
	 
	 signal section_total : integer := 0;
	 signal section_counter : integer := 0;
	 signal step_counter : integer := 0;
	 signal element_counter : integer := 0;
	 signal base_addr : integer := 0;
	 
    signal addr1_int : integer range 0 to N/2 ;
    signal state : state_ntt;
    signal layer_int : integer range 0 to log2n-1;
	 signal done_int : std_logic := '0';
    signal diff : integer range 0 to N/2;
	 signal toggle : std_logic := '0';
begin
    diff <= N/(2**(layer_int+2));
	 section_total <= 2*diff;
	 -- also need internal counter for gap that will appear
	 -- when element and section count are equal we can move on 
	 -- we can use a shift register to delay values as much as we need 
    process(clk,reset)
        begin
        if reset = '1' then
				step_counter <= 0;
				--step_address <= '0';
            addr1_int <= 0;
            state <= reset_state;
        elsif rising_edge(clk) then
            case state is
                -- toggle 
					-- if toggle  == '0' or 1
					 -- small - addr1_int <= step_address; toggle <= 1;
					 -- large - addr1_int <= step; toggle <= 1
                when reset_state =>
                    layer_int <= 0; -- may want to change to 1 when we actually do this since data will be input in bit reverse order
                    state <= compute_state;
               
                when compute_state =>
						if element_counter < count_total then
							 if section_counter = section_total then
								  base_addr <= element_counter;
								  step_counter <= 0;
								  section_counter <= 0;
								  --base_addr <= section_total;
								  --toggle <= '0';
							 else
								  if toggle = '0' then
										addr1_int <= base_addr + step_counter;
										toggle <= '1';
								  else
										addr1_int <= base_addr + diff + step_counter;
										step_counter <= step_counter + 1; -- Only increment in this path
										toggle <= '0';
								  end if;

								  section_counter <= section_counter + 1;
								  element_counter <= element_counter + 1;
							 end if;
						
                elsif layer_int < log2n - 2 then
                    layer_int <= layer_int + 1;
						  addr1_int <= 0;
						  
						  element_counter <= 0;
						  step_counter <= 0;
						  section_counter <= 0;
						  base_addr <= 0;
                    state <= compute_state;
                else
                    state <= done_state;
                end if;
             
               
               when done_state =>
						  done_int <= '1';
                    state <= done_state;
               
        end case;
       
        addr_1 <= std_logic_vector( to_unsigned(addr1_int, depth) );
        --addr_2 <= std_logic_vector( to_unsigned(addr1_int + diff, depth) );
        layer <= layer_int;
        --step_o <= step_counter;
		  done <= done_int;	
		  --count <= step_counter;
		  step_o <= 0;
		  count1 <= section_counter;
		  count2 <= element_counter;
        end if;
    end process;
   
end architecture;