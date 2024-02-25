library ieee;
use ieee.std_logic_1164.all;

entity twobit is
	port(
		clk : in std_logic;
		data_in : in std_logic_vector(1 downto 0);
		data_out : out std_logic_vector(1 downto 0);
		writeread : in std_logic --1 equals write, 0 equals read	
		);
	end twobit;

architecture Behavioral of twobit is 	
begin
	signal b : std_logic_vector(1 down to 0) := "00";
   process(clk)
	 variable a : std_logic_vector(1 down to 0) := "00";
	begin 
		if rising_edge(clk) then
			if writeread = '1' then
				a := data_in;
			elsif writeread = '0' then
				b <= a;
			end if;
		end if;
	end process;
	data_out <= b;
end architecture;
		
				
		
			
			
