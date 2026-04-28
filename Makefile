.PHONY: run-lab1 run-lab1-s run-lab2 run-lab2-s run-lab3 run-lab3-s run-lab4 run-lab4-s run-lab5-s run-lab6-s bench-lab1 bench-lab1-s bench-lab2 bench-lab2-s bench-lab3 bench-lab3-s bench-lab4 bench-lab4-s bench-lab5-s bench-lab6-s bench-prefix-lab3-s bench-prefix-lab4-s

PYTHON := .venv/bin/python

run-lab1:
	$(PYTHON) example.py --lab 1

run-lab1-s:
	$(PYTHON) example.py --lab 1 --solution

run-lab2:
	$(PYTHON) example.py --lab 2

run-lab2-s:
	$(PYTHON) example.py --lab 2 --solution

run-lab3:
	$(PYTHON) example.py --lab 3

run-lab3-s:
	$(PYTHON) example.py --lab 3 --solution

run-lab4:
	$(PYTHON) example.py --lab 4

run-lab4-s:
	$(PYTHON) example.py --lab 4 --solution

run-lab5-s:
	$(PYTHON) example.py --lab 5 --solution

run-lab6-s:
	$(PYTHON) example.py --lab 6 --solution --data-parallel-size 2

bench-lab1:
	$(PYTHON) bench.py --lab 1

bench-lab1-s:
	$(PYTHON) bench.py --lab 1 --solution

bench-lab2:
	$(PYTHON) bench.py --lab 2

bench-lab2-s:
	$(PYTHON) bench.py --lab 2 --solution

bench-lab3:
	$(PYTHON) bench.py --lab 3

bench-lab3-s:
	$(PYTHON) bench.py --lab 3 --solution

bench-lab4:
	$(PYTHON) bench.py --lab 4

bench-lab4-s:
	$(PYTHON) bench.py --lab 4 --solution

bench-lab5-s:
	$(PYTHON) bench.py --lab 5 --solution

bench-lab6-s:
	$(PYTHON) bench.py --lab 6 --solution --data-parallel-size 2

bench-prefix-lab3-s:
	$(PYTHON) bench_prefix.py --lab 3 --solution

bench-prefix-lab4-s:
	$(PYTHON) bench_prefix.py --lab 4 --solution
