library ieee;
use ieee.std_logic_1164.all;

-- Define the entity for the shift register
entity shift_register is
    generic (
        REGISTER_DEPTH : integer := 4096 -- 4096 elements
    );
    port (
        clk : in std_logic; -- Clock input
		  write_read : in std_logic 
        data_in : in std_logic_vector(1 downto 0); -- Input data
        data_out : out std_logic_vector(1 downto 0) -- Output data
    );
end shift_register;

-- Define the architecture for the shift register
architecture Behavioral of shift_register is

-- declare twobit ports
    component twobit is 
        port (
            clk, writeread : in std_logic;
            datain : in std_logic_vector(1 downto 0);
            dataout : out std_logic_vector(1 downto 0)
        );   
	  end component;
begin
    process(clk, reset)
        variable i : integer;
    begin
        if rising_edge(clk) then -- Shift on rising clock edge
                for i in REGISTER_DEPTH - 1 generate 
						twobit_inst : twobit 
						port map 
						(  datain => data_in, 
							dataout => data_out, 
							writeread => write_read, 
							clk => clk
						);
                end loop;
						-- new for loop where we input data into first register and then pass data along, possibly pass 00s
					for i in REGISTER_DEPTH - 1 
						-- code 
					end loop;
        end if;
    end process;

    -- Output the data from the last element
    data_out <= register(REGISTER_DEPTH - 1);
	 -- change this to use new scheme

end Behavioral;
