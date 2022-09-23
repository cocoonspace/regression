package regression

import (
	"errors"
	"math"

	"gonum.org/v1/gonum/mat"
)

var (
	// ErrNotEnoughData signals that there weren't enough datapoint to train the model.
	ErrNotEnoughData = errors.New("not enough data points")
	// ErrTooManyVars signals that there are too many variables for the number of observations being made.
	ErrTooManyVars = errors.New("not enough observations to support this many variables")
	// ErrRegressionRun signals that the Run method has not been run yet.
	ErrRegressionRun = errors.New("regression has not run yet")
)

// Regression is the exposed data structure for interacting with the API.
type Regression struct {
	Data              []DataPoint
	coeff             map[int]float64
	R2                float64
	VarianceObserved  float64
	VariancePredicted float64
	initialised       bool
	crosses           []featureCross
	Ready             bool
}

type DataPoint struct {
	Observed  float64
	Variables []float64
	Crosses   []float64
	Predicted float64
	Error     float64
}

// DataPoints is a slice of DataPoint
// This type allows for easier construction of training data points.
type DataPoints []DataPoint

// Predict updates the "Predicted" value for the inputed features.
func (r *Regression) Predict(vars []float64) (float64, error) {
	if !r.Ready {
		return 0, ErrRegressionRun
	}

	// apply any features crosses to vars
	for _, cross := range r.crosses {
		vars = append(vars, cross.Calculate(vars)...)
	}

	p := r.Coeff(0)
	for j, val := range vars {
		p += r.Coeff(j+1) * val
	}
	return p, nil
}

// AddCross registers a feature cross to be applied to the data points.
func (r *Regression) AddCross(cross featureCross) {
	r.crosses = append(r.crosses, cross)
}

// Train the regression with some data points.
func (r *Regression) Train(d ...DataPoint) {
	r.Data = append(r.Data, d...)
	if len(r.Data) > 2 {
		r.initialised = true
	}
}

// Apply any feature crosses, generating new observations and updating the data points, as well as
// populating variable names for the feature crosses.
func (r *Regression) applyCrosses() {
	if len(r.crosses) == 0 {
		return
	}
	for _, p := range r.Data {
		if len(p.Crosses) > 0 {
			continue
		}
		for _, c := range r.crosses {
			p.Crosses = c.Calculate(p.Variables)
		}
	}
}

// Run determines if there is enough data present to run the regression
// and whether or not the training has already been completed.
// Once the above checks have passed feature crosses are applied if any
// and the model is trained using QR decomposition.
func (r *Regression) Run() error {
	if !r.initialised {
		return ErrNotEnoughData
	}

	// apply any features crosses
	r.applyCrosses()
	r.Ready = true

	observations := len(r.Data)
	numOfvars := len(r.Data[0].Variables) + len(r.Data[0].Crosses)

	if observations < (numOfvars + 1) {
		return ErrTooManyVars
	}

	// Create some blank variable space
	observed := mat.NewDense(observations, 1, nil)
	variables := mat.NewDense(observations, numOfvars+1, nil)

	for i := 0; i < observations; i++ {
		observed.Set(i, 0, r.Data[i].Observed)
		variables.Set(i, 0, 1)
		for j, val := range r.Data[i].Variables {
			variables.Set(i, j+1, val)
		}
		for j, val := range r.Data[i].Crosses {
			variables.Set(i, len(r.Data[i].Variables)+j, val)
		}
	}

	// Now run the regression
	_, n := variables.Dims() // cols
	qr := new(mat.QR)
	qr.Factorize(variables)
	q := new(mat.Dense)
	reg := new(mat.Dense)
	qr.QTo(q)
	qr.RTo(reg)

	qtr := q.T()
	qty := new(mat.Dense)
	qty.Mul(qtr, observed)

	c := make([]float64, n)
	for i := n - 1; i >= 0; i-- {
		c[i] = qty.At(i, 0)
		for j := i + 1; j < n; j++ {
			c[i] -= c[j] * reg.At(i, j)
		}
		c[i] /= reg.At(i, i)
	}

	// Output the regression results
	r.coeff = make(map[int]float64, numOfvars)
	for i, val := range c {
		r.coeff[i] = val
	}

	r.calcPredicted()
	r.calcVariance()
	r.calcR2()
	return nil
}

// Coeff returns the calculated coefficient for variable i.
func (r *Regression) Coeff(i int) float64 {
	if len(r.coeff) == 0 {
		return 0
	}
	return r.coeff[i]
}

// GetCoeffs returns the calculated coefficients. The element at index 0 is the offset.
func (r *Regression) GetCoeffs() []float64 {
	if len(r.coeff) == 0 {
		return nil
	}
	coeffs := make([]float64, len(r.coeff))
	for i := range coeffs {
		coeffs[i] = r.coeff[i]
	}
	return coeffs
}

func (r *Regression) calcPredicted() {
	observations := len(r.Data)
	for i := 0; i < observations; i++ {
		r.Data[i].Predicted, _ = r.Predict(r.Data[i].Variables)
		r.Data[i].Error = r.Data[i].Predicted - r.Data[i].Observed
	}
}

func (r *Regression) calcVariance() {
	observations := len(r.Data)
	var obtotal, prtotal, obvar, prvar float64
	for i := 0; i < observations; i++ {
		obtotal += r.Data[i].Observed
		prtotal += r.Data[i].Predicted
	}
	obaverage := obtotal / float64(observations)
	praverage := prtotal / float64(observations)

	for i := 0; i < observations; i++ {
		obvar += math.Pow(r.Data[i].Observed-obaverage, 2)
		prvar += math.Pow(r.Data[i].Predicted-praverage, 2)
	}
	r.VarianceObserved = obvar / float64(observations)
	r.VariancePredicted = prvar / float64(observations)
}

func (r *Regression) calcR2() {
	r.R2 = r.VariancePredicted / r.VarianceObserved
}

// MakeDataPoints makes a `[]DataPoint` from a `[][]float64`. The expected fomat for the input is a row-major [][]float64.
// That is to say the first slice represents a row, and the second represents the cols.
// Furthermore it is expected that all the col slices are of the same length.
// The obsIndex parameter indicates which column should be used
func MakeDataPoints(a [][]float64, obsIndex int) []DataPoint {
	if obsIndex != 0 && obsIndex != len(a[0])-1 {
		return perverseMakeDataPoints(a, obsIndex)
	}

	retVal := make([]DataPoint, 0, len(a))
	if obsIndex == 0 {
		for _, r := range a {
			retVal = append(retVal, DataPoint{Observed: r[0], Variables: r[1:]})
		}
		return retVal
	}

	// otherwise the observation is expected to be the last col
	last := len(a[0]) - 1
	for _, r := range a {
		retVal = append(retVal, DataPoint{Observed: r[last], Variables: r[:last]})
	}
	return retVal
}

func perverseMakeDataPoints(a [][]float64, obsIndex int) []DataPoint {
	retVal := make([]DataPoint, 0, len(a))
	for _, r := range a {
		obs := r[obsIndex]
		others := make([]float64, 0, len(r)-1)
		for i, c := range r {
			if i == obsIndex {
				continue
			}
			others = append(others, c)
		}
		retVal = append(retVal, DataPoint{Observed: obs, Variables: others})
	}
	return retVal
}
