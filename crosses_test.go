package regression

import (
	"testing"
)

func TestPowCrosses(t *testing.T) {
	cross1 := PowCross(0, 2) // cross of the variable at index 0
	if cross1.Calculate([]float64{2})[0] != 4 {
		t.Error("Incorrect value")
	}

	cross2 := PowCross(1, 2) // cross of the variable at index 1
	if cross2.Calculate([]float64{2, -3})[0] != 9 {
		t.Error("Incorrect value, got", cross2.Calculate([]float64{2, -3}))
	}
}

func TestMultiplicationCrosses(t *testing.T) {
	cross1 := MultiplierCross(0, 1, 3)
	if cross1.Calculate([]float64{2, 3, 4, 5})[0] != 30 {
		t.Errorf("Incorrect value, expected 30 got %.2f", cross1.Calculate([]float64{2, 3, 4, 5})[0])
	}

	cross2 := MultiplierCross(0, 1)
	if cross2.Calculate([]float64{2, 3})[0] != 6 {
		t.Errorf("Incorrect value, expected 6 got %.2f", cross1.Calculate([]float64{2, 3, 4, 5})[0])
	}
}
