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
    -- Declare twobit component
    component twobit is 
        port (
            clk, writeread : in std_logic;
            datain : in std_logic_vector(1 downto 0);
            dataout : out std_logic_vector(1 downto 0)
        );   
    end component;

begin
    process(clk)
    begin
        if rising_edge(clk) then -- Shift on rising clock edge
            if write_read = '1' then -- Only perform shift if write_read signal is high
                for i in 0 to REGISTER_DEPTH - 2 generate
                    -- Pass data through the twobit components
                    twobit_inst_i : twobit port map (
                        clk => clk,
                        writeread => write_read,
                        datain => data_in,
                        dataout => data_out
                    );
                end generate;
            end if;
        end if;
    end process;

end Behavioral;
