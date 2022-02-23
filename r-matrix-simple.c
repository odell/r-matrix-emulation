#include <stdio.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include <gsl/gsl_sf.h>

#define HBARC 197.3269804 // MeV•fm
#define MASS_TRITON 2808.92113298 // MeV
#define MASS_DEUTERON 1875.61294257 // MeV
#define MASS_NEUTRON 939.565420 // MeV
#define MASS_ALPHA 3727.37940 // MeV
const double MU_D = MASS_TRITON*MASS_DEUTERON/(MASS_TRITON + MASS_DEUTERON);
const double MU_N = MASS_ALPHA*MASS_NEUTRON/(MASS_ALPHA + MASS_NEUTRON);
const double ALPHA = 1.0/137.035999084;
const double Q = 17.6;
const double J = 1.5;
const double J1 = 0.5;
const double J2 = 1.0;

double relative_momentum(double e, double mu)
{
  return sqrt(2*mu*e)/HBARC;
}

double sommerfeld(double e, double mu)
{
  return ALPHA * sqrt(mu/(2*e));
}

void coulomb_functions(double* F, double* Fp, double* G, double* Gp, int l, double eta,
    double x)
{
  int info;
  double exp_f, exp_g;
  gsl_sf_result f, fp, g, gp;
  info = gsl_sf_coulomb_wave_FG_e(eta, x, l, 0, &f, &fp, &g, &gp, &exp_f, &exp_g);
  // printf("%.4f  %.4f\n", exp_f, exp_g);
  if (info == GSL_EOVRFLW) printf("overflow or underflow encountered\n");
  *F = f.val*exp(exp_f);
  *Fp = fp.val*exp(exp_f);
  *G = g.val*exp(exp_g);
  *Gp = gp.val*exp(exp_g);
}

double penetration_factor(int l, double eta, double x)
{
  double F, Fp, G, Gp;
  coulomb_functions(&F, &Fp, &G, &Gp, l, eta, x);
  return x / (F*F + G*G);
}

double shift_factor(int l, double eta, double x)
{
  double F, Fp, G, Gp;
  coulomb_functions(&F, &Fp, &G, &Gp, l, eta, x);
  return x * (F*Fp + G*Gp) / (F*F + G*G);
}

double Gamma_c(double gamma_c_sq, int l, double eta, double x)
{
  return 2 * gamma_c_sq * penetration_factor(l, eta, x);
}

double Delta_c(double gamma_c_2, double k, double r, double Bc, int l, double eta)
{
	return -gamma_c_2 * (shift_factor(l, eta, k*r)-Bc);
}

double Sdn2(double energy, double e0, double Bd, double Bn, double gamma_d_2,
    double gamma_n_2, double kd, double ad, double kn, double an, double eta,
    double A)
{
	double Gamma_d = Gamma_c(gamma_d_2, 0, eta, kd*ad);
	double Gamma_n = Gamma_c(gamma_n_2, 2, 0, kn*an);
	double Gamma = Gamma_d + Gamma_n;
	double Delta_d = Delta_c(gamma_d_2, kd, ad, Bd, 0, eta);
	double Delta_n = Delta_c(gamma_n_2, kn, an, Bn, 2, 0);
	double Delta = Delta_d + Delta_n;
  // printf("Sdn2: %.2e  %.2e  %.2e\n", Gamma_d, Gamma_n, Delta);
	return Gamma_d*Gamma_n / ((e0+Delta-energy)*(e0+Delta-energy) + Gamma*Gamma/4) + 
	       2.0/M_PI*A * penetration_factor(0, eta, kd*ad) * penetration_factor(0, 0.0, kn*an);
}

/*
Evaluates the cross section (barn = 100fm^2) at energy en (MeV) with R-matrix parameters:
		eB (MeV) where Bc = shift_factor(eB),
		gamma_d_2 (MeV) is the reduced deuteron width,
		gamma_n_2 (MeV) is the reduced neutron width,
		ad (fm) is the deuteron channel radius,
		an (fm) is the neutron channel radius,
and ue (MeV) is the electron screening potential.
*/
double cross_section(double en, double e0, double eB, double gamma_d_2,
		double gamma_n_2, double ad, double an, double ue, double A)
{
	if (en < 0.0 || e0 < 0.0 || eB < 0.0 || gamma_d_2 < 0.0 ||
			gamma_n_2 < 0.0 || ad < 0.0 || an < 0.0 || ue < 0.0) {
		return INFINITY;
	}
	double kd = relative_momentum(en, MU_D); // 1/fm
	double kn = relative_momentum(en+Q, MU_N); // 1/fm
	double kdB = relative_momentum(eB, MU_D); // 1/fm
	double knB = relative_momentum(eB+Q, MU_N); // 1/fm
	double eta = sommerfeld(en, MU_D); // dimensionless
	double etaB = sommerfeld(eB, MU_D); // dimensionless
  // printf("kdB = %.2e\n", kdB);
	double Bd = shift_factor(0, etaB, kdB*ad); // dimensionless
	double Bn = shift_factor(2, 0, knB*an); // dimensionless
  // printf("Bd = %.2e  kdB = %.2e  ad = %.2e\n", Bd, kdB, ad);
	return exp(M_PI*eta*ue/en) * 
			M_PI/(kd*kd) * (2.0*J+1.0)/((2.0*J1+1.0)*(2.0*J2+1.0)) *
			Sdn2(en, e0, Bd, Bn, gamma_d_2, gamma_n_2, kd, ad, kn, an, eta, A) / 100;
}

/*
Evaluates the S-factor (MeV•barn = MeV•100fm^2) at energy en (MeV) with R-matrix parameters:
		eB (MeV) where Bc = shift_factor(eB),
		gamma_d_2 (MeV) is the reduced deuteron width,
		gamma_n_2 (MeV) is the reduced neutron width,
		ad (fm) is the deuteron channel radius,
		an (fm) is the neutron channel radius,
and ue (MeV) is the electron screening potential.
*/
double S_factor(double en, double e0, double eB, double gamma_d_2,
		double gamma_n_2, double ad, double an, double ue, double A)
{
	if (en < 0.0 || e0 < 0.0 || eB < 0.0 || gamma_d_2 < 0.0 ||
			gamma_n_2 < 0.0 || ad < 0.0 || an < 0.0 || ue < 0.0) {
		return INFINITY;
	}
	double kd = relative_momentum(en, MU_D); // 1/fm
	double kn = relative_momentum(en+Q, MU_N); // 1/fm
	double kdB = relative_momentum(eB, MU_D); // 1/fm
	double knB = relative_momentum(eB+Q, MU_N); // 1/fm
	double eta = sommerfeld(en, MU_D); // dimensionless
	double etaB = sommerfeld(eB, MU_D); // dimensionless
  // printf("kdB = %.2e\n", kdB);
	double Bd = shift_factor(0, etaB, kdB*ad); // dimensionless
	double Bn = shift_factor(2, 0, knB*an); // dimensionless
  // printf("Bd = %.2e  kdB = %.2e  ad = %.2e\n", Bd, kdB, ad);
	return exp(M_PI*eta*ue/en) * en * exp(2*M_PI*eta) *\
			M_PI/(kd*kd) * (2.0*J+1.0)/((2.0*J1+1.0)*(2.0*J2+1.0)) *\
			Sdn2(en, e0, Bd, Bn, gamma_d_2, gamma_n_2, kd, ad, kn, an, eta, A) / 100;
}

double S_factor_unitary_limit(double en, double ue)
{
	if (en < 0.0 || ue < 0.0) {
		return INFINITY;
	}
	double kd = relative_momentum(en, MU_D); // 1/fm
	double eta = sommerfeld(en, MU_D); // dimensionless
	return exp(M_PI*eta*ue/en) * en * exp(2*M_PI*eta) *\
			M_PI/(kd*kd) * (2.0*J+1.0)/((2.0*J1+1.0)*(2.0*J2+1.0)) * 1.0 / 100;
}

