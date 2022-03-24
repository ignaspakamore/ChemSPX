def print_logo():


	print("""
  _____  _     _  ______ _______      _______  _____  _     _
 |       |_____| |______ |  |  |      |______ |_____]  \___/ 
 |_____  |     | |______ |  |  |      ______| |       _/   \_
_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_                                                            
                                                             """)
	print(f"{18*' '}Chemical Space Explorer")
	print(f"{18*' '}     I. Pakamore       ")
	print(f"{19*' '}University of Glasgow\n")

	
def print_pars(indict, *args):
	print('=====================PARAMETERS===========================')
	[print(val, "=", key) for val, key in indict.items()]
	print('==========================================================')
	#extra info
	[print(i) for i in args]
	print("\n")


def print_finished():
	print("""
____ _ _  _ _ ____ _  _ ____ ___  
|___ | |\ | | [__  |__| |___ |  \ 
|    | | \| | ___] |  | |___ |__/ 
                                """)

def print_loop_info(step, mean_fx, mean_derr_fx, av_vect_change, loop_time, method, print_every):
	if step == 1:
		print(f"""\n{20*" "}OPTIMISATION -- {method} METHOD\n""")
		print("{:<11s}{:<11s}{:>10s}{:>15s}{:>22s}".format('Step', '<f(x)>', '\u0394<f(x)>', '<\u0394VEC>', 'Loop time(s)'))

		print(f"""{70*"-"}""")
		print("{:<8d}{:<15e}{:>10e}{:>16e}{:>15f}".format(step, mean_fx, mean_derr_fx,av_vect_change, loop_time))
	elif step % int(print_every) == 0:
		print("{:<8d}{:<15e}{:>10e}{:>16e}{:>15f}".format(step, mean_fx, mean_derr_fx,av_vect_change, loop_time))

def init_info( mean_fx, std_fx, pop_size):
	
	print(f"""INITIAL DATA""")
	print(f"""{30*"-"}""")
	print(f"""mean f(x)       = {mean_fx:.5f}\nSTD  f(x)       = {std_fx:.5f}\nsample size = {pop_size}""")
	

def print_void_info(step, n_neighbours, loop_time):
	if step == 1:
		print(f"""\n{13*" "}VOID SEARCH -- GA\n""")
		print("{:<10s}{:>10s}{:>20s}".format('DP', 'N-neighbours', 'Loop time(s)'))
		print(f"""{50*"-"}""")
		print("{:<10d}{:>5d}{:>25f}".format(step, n_neighbours, loop_time))
	else:
		print("{:<10d}{:>5d}{:>25f}".format(step, n_neighbours, loop_time))

def print_loop_conv(par1='YES', par2='YES', par3='YES'):
	print(f"""\n{22*" "}***CONVERGENCE***""")
	#print("{:<11s}{:<11s}{:>10s}{:>15s}{:>22s}".format(' ', '<f(x)>', '\u0394<f(x)>', '<VEC>', ' '))
	print("{:<13s}{:<15s}{:>1s}{:>16s}{:>15s}\n".format(' ', par1, par2, par3, ' '))
























