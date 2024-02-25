library ieee;
use ieee.std.logic.1164;

-- Define the entity for the shift register
entity shift_register is
    generic (
        DATA_WIDTH : integer := 2; -- 2 bits per element
        REGISTER_DEPTH : integer := 4096 -- 4096 elements
    );
    port (
        clk : in std_logic; -- Clock input
        reset : in std_logic; -- Reset input
        data_in : in std_logic_vector(DATA_WIDTH - 1 downto 0); -- Input data
        data_out : out std_logic_vector(DATA_WIDTH - 1 downto 0) -- Output data
    );
end shift_register;

-- Define the architecture for the shift register
architecture Behavioral of shift_register is
    type register_array is array (0 to REGISTER_DEPTH - 1) of std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal register : register_array := (others => (others => '0')); -- Initialize all elements to '0'
begin
    process(clk, reset)
        variable i : integer;
    begin
        if reset = '1' then -- Synchronous reset
            register <= (others => (others => '0')); -- Reset all elements to '0'
        elsif rising_edge(clk) then -- Shift on rising clock edge
            if (data_in'length = DATA_WIDTH) then
                -- Shift the data
                for i in REGISTER_DEPTH - 2 downto 0 loop
                    register(i + 1) <= register(i);
                end loop;
                register(0) <= data_in; -- Input data enters at the first element
            end if;
        end if;
    end process;

    -- Output the data from the last element
    data_out <= register(REGISTER_DEPTH - 1);

end Behavioral;
