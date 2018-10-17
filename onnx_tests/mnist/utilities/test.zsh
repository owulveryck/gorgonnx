#!/bin/zsh
#imgcat <(go run decode_input.go ../test_data_set_1/input_0.pb ../test_data_set_1/output_0.pb 3>&1 1>&2 2>&3)
imgcat <(go run decode_input.go ../test_data_set_1/input_0.pb ../test_data_set_1/output_0.pb)
