bin_dir=../release/timing

factor: FORCE
#	${bin_dir}/factor/table_timing --num-values 2 --max-arity 10 --num-reps 50000
#	${bin_dir}/factor/table_timing --num-values 5 --max-arity 8 --num-reps 1000
#	${bin_dir}/factor/vector_timing --step-size 10 --max-values 101 --num-reps=100000
	${bin_dir}/factor/matrix_timing --step-size 10 --max-values 32 --num-reps=100000

factor-experimental: FORCE
#	${bin_dir}/factor/experimental/table_experimental_timing --num-values 2 --max-arity 9 --num-reps 50000
#	${bin_dir}/factor/experimental/table_experimental_timing --num-values 5 --max-arity 7 --num-reps 1000
#	${bin_dir}/factor/experimental/vector_experimental_timing --step-size 10 --max-values 101 --num-reps=100000
	${bin_dir}/factor/experimental/matrix_experimental_timing --step-size 10 --max-values 32 --num-reps=100000


FORCE:
