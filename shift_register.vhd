library ieee;
use ieee.std_logic_1164.all;

-- Define the entity for the shift register
entity shift_register is
    generic (
        REGISTER_DEPTH : integer := 4096 -- 4096 elements
    );
    port (
        clk : in std_logic; -- Clock input
        write_read : in std_logic;
        data_in : in std_logic_vector(1 downto 0); -- Input data
        data_out : out std_logic_vector(1 downto 0) -- Output data
    );
end shift_register;

-- Define the architecture for the shift register
architecture Behavioral of shift_register is
    type twobit_array is array (0 to REGISTER_DEPTH - 1) of std_logic_vector(1 downto 0); -- Change to std_logic_vector

    -- Declare twobit ports
    component twobit is 
        port (
            clk, writeread : in std_logic;
            datain : in std_logic_vector(1 downto 0);
            dataout : out std_logic_vector(1 downto 0)
        );   
    end component;

    -- Declare shift register signal
    signal shift_reg : twobit_array;

begin
    process(clk)
        variable i : integer;
    begin
        if rising_edge(clk) then -- Shift on rising clock edge
            if write_read = '1' then -- Only perform shift if write_read signal is high
                -- Shift data through the shift register
                for i in REGISTER_DEPTH - 2 downto 0 loop
                    shift_reg(i + 1) <= shift_reg(i);
                end loop;
                
                -- Input data into the first register
                shift_reg(0) <= data_in;
            end if;
        end if;
    end process;

    -- Output the data from the last element
    data_out <= shift_reg(REGISTER_DEPTH - 1);

end Behavioral;
