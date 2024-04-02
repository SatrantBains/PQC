library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.STD_LOGIC_ARITH.ALL;
use IEEE.STD_LOGIC_UNSIGNED.ALL;

entity SignInverter is
    Port (
        clk : in STD_LOGIC;
        rst : in STD_LOGIC; -- Reset signal (active high)
        input_val : in STD_LOGIC_VECTOR(1 downto 0); -- 2-bit input
        output_val : out STD_LOGIC_VECTOR(1 downto 0) -- 2-bit output with inverted sign
    );
end SignInverter;

architecture Behavioral of SignInverter is
    signal temp_val : STD_LOGIC_VECTOR(1 downto 0);
begin
    process(clk, rst)
    begin
        if rst = '1' then
            -- Reset the output to 0 when reset is high
            temp_val <= (others => '0');
        elsif rising_edge(clk) then
            -- Invert the sign bit (MSB) on each clock's rising edge
            temp_val <= input_val;
            temp_val(1) <= NOT input_val(1);
        end if;
    end process;
    
    -- Assign the internal signal to the output
    output_val <= temp_val;
end Behavioral;
