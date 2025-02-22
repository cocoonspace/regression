package regression

import (
	"math"
	"strconv"
)

type featureCross interface {
	Calculate([]float64) []float64 // must return the same number of features each run
}

type functionalCross struct {
	boundVars []int
	crossFn   func([]float64) []float64
}

func (c *functionalCross) Calculate(input []float64) []float64 {
	return c.crossFn(input)
}

// Feature cross based on computing the power of an input.
func PowCross(i int, power float64) featureCross {
	return &functionalCross{
		boundVars: []int{i},
		crossFn: func(vars []float64) []float64 {
			return []float64{math.Pow(vars[i], power)}
		},
	}
}

// Feature cross based on the multiplication of multiple inputs.
func MultiplierCross(vars ...int) featureCross {
	name := ""
	for i, v := range vars {
		name += strconv.Itoa(v)
		if i < (len(vars) - 1) {
			name += "*"
		}
	}

	return &functionalCross{
		boundVars: vars,
		crossFn: func(input []float64) []float64 {
			var output float64 = 1
			for _, variableIndex := range vars {
				output *= input[variableIndex]
			}
			return []float64{output}
		},
	}
}
