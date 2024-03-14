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

    -- Declare signal for the shift register
    signal shift_reg : std_logic_vector(1 downto 0) := (others => '0');

begin
    process(clk, reset)
        variable i : integer;
    begin
        if rising_edge(clk) then -- Shift on rising clock edge
            if write_read = '1' then -- Only perform shift if write_read signal is high
                -- Shift data through the register
                for i in REGISTER_DEPTH - 2 downto 0 loop
                    shift_reg(i) => shift_reg(i+1);
                end loop;
                
                -- Input data into the first register
                shift_reg(0) <= data_in;
            end if;
        end if;
    end process;

    -- Instantiate twobit components
    twobit_inst : for i in 0 to REGISTER_DEPTH - 1 generate
        twobit_inst_i : twobit port map (
            clk => clk,
            writeread => write_read,
            datain => shift_reg(i),
            dataout => shift_reg(i + 1)
        );
    end generate;

    -- Output the data from the last element
    data_out <= shift_reg(REGISTER_DEPTH - 1);

end Behavioral;
