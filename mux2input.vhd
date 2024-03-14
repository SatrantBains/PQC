library ieee;
use ieee.std_logic_1164.all;

entity mux2input is
port (
	input0 : in std_logic_vector(1 downto 0);
	input1 : in std_logic_vector(1 downto 0);
	sig : in std_logic; --placeholder
	muxoutput : out std_logic_vector(1 downto 0)
	);
end mux2input;

architecture behavioral of mux2input is
begin

	process(input0, input1, sig)
	begin 
		if sig = '0' then
			muxoutput <= input0;
		else
			muxoutput <= input1;
		end if;
	end process;
end behavioral;

