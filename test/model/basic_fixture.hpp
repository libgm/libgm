#ifndef LIBGM_TEST_MODEL_BASIC_FIXTURE_HPP
#define LIBGM_TEST_MODEL_BASIC_FIXTURE_HPP

#include <vector>

#include <libgm/argument/universe.hpp>
#include <libgm/factor/probability_table.hpp>

using namespace libgm;

struct basic_fixture {

  basic_fixture() {
    history = u.new_finite_variable("history", history_arity);
    cvp = u.new_finite_variable("cvp", cvp_arity);
    pcwp = u.new_finite_variable("pcwp", pcwp_arity);
    hypovolemia = u.new_finite_variable("hypovolemia", hypovolemia_arity);
    lvedvolume = u.new_finite_variable("lvedvolume", lvedvolume_arity);
    lvfailure = u.new_finite_variable("lvfailure", lvfailure_arity);
    strokevolume = u.new_finite_variable("strokevolume", strokevolume_arity);
    errlowoutput = u.new_finite_variable("errlowoutput", errlowoutput_arity);
    hrbp = u.new_finite_variable("hrbp", hrbp_arity);
    hrekg = u.new_finite_variable("hrekg", hrekg_arity);
    errcauter = u.new_finite_variable("errcauter", errcauter_arity);
    hrsat = u.new_finite_variable("hrsat", hrsat_arity);
    insuffanesth = u.new_finite_variable("insuffanesth", insuffanesth_arity);
    anaphylaxis = u.new_finite_variable("anaphylaxis", anaphylaxis_arity);
    tpr = u.new_finite_variable("tpr", tpr_arity);
    expco2 = u.new_finite_variable("expco2", expco2_arity);
    kinkedtube = u.new_finite_variable("kinkedtube", kinkedtube_arity);
    minvol = u.new_finite_variable("minvol", minvol_arity);
    fio2 = u.new_finite_variable("fio2", fio2_arity);
    pvsat = u.new_finite_variable("pvsat", pvsat_arity);
    sao2 = u.new_finite_variable("sao2", sao2_arity);
    pap = u.new_finite_variable("pap", pap_arity);
    pulmembolus = u.new_finite_variable("pulmebolus", pulmembolus_arity);
    shunt = u.new_finite_variable("shunt", shunt_arity);
    intubation = u.new_finite_variable("intubation", intubation_arity);
    press = u.new_finite_variable("press", press_arity);
    disconnect = u.new_finite_variable("disconnect", disconnect_arity);
    minvolset = u.new_finite_variable("minvolset", minvolset_arity);
    ventmach = u.new_finite_variable("ventmach", ventmach_arity);
    venttube = u.new_finite_variable("venttube", venttube_arity);
    ventlung = u.new_finite_variable("ventlung", ventlung_arity);
    ventalv = u.new_finite_variable("ventalv", ventalv_arity);
    artco2 = u.new_finite_variable("artco2", artco2_arity);
    catechol = u.new_finite_variable("catechol", catechol_arity);
    hr = u.new_finite_variable("hr", hr_arity);
    co = u.new_finite_variable("co", co_arity);
    bp = u.new_finite_variable("bp", bp_arity);

    // P(HISTORY | LVFAILURE)
    factors.push_back(ptable({lvfailure, history}, 1.0));
    // P(CVP | LVEDVOLUME)
    factors.push_back(ptable({lvedvolume, cvp}, 1.0));
    // P(PCWP | LVEDVOLUME)
    factors.push_back(ptable({lvedvolume, pcwp}, 1.0));
    // P(HYPOVOLEMIA)
    factors.push_back(ptable({hypovolemia}, 1.0));
    // P(LVEDVOLUME | HYPOVOLEMIA, LVFAILURE)
    factors.push_back(ptable({hypovolemia, lvfailure, lvedvolume}, 1.0));
    // P(LVFAILURE)
    factors.push_back(ptable({lvfailure}, 1.0));
    // P(STROKEVOLUME | HYPOVOLEMIA, LVFAILURE)
    factors.push_back(ptable({hypovolemia, lvfailure, strokevolume}, 1.0));
    // P(ERRLOWOUTPUT)
    factors.push_back(ptable({errlowoutput}, 1.0));
    // P(HRBP | ERRLOWOUTPUT, HR)
    factors.push_back(ptable({errlowoutput, hr, hrbp}, 1.0));
  }

  // Variables
  // =========================================================================

  // Note: the last constant in each enum is the arity of the variable
  universe u;

  // Variable: history (0)
  enum history_values { history_true,
                        history_false,
                        history_arity };

  // Variable: CVP (1)
  enum cvp_values { cvp_low,
                    cvp_normal,
                    cvp_high,
                    cvp_arity };

  // Variable: PCWP (2)
  enum pcwp_values { pcwp_low,
                     pcwp_normal,
                     pcwp_high,
                     pcwp_arity };

  // Variable: hypovolemia (3)
  enum hypovolemia_values { hypovolemia_true,
                            hypovolemia_false,
                            hypovolemia_arity };

  // Variable: LVED volume (4)
  enum lvedvolume_values { lvedvolume_low,
                           lvedvolume_normal,
                           lvedvolume_high,
                           lvedvolume_arity };

  // Variable: liver failure (5)
  enum lvfailure_values { lvfailure_true,
                          lvfailure_false,
                          lvfailure_arity };

  // Variable: stroke volume (6)
  enum strokevolume_values { strokevolume_low,
                             strokevolume_normal,
                             strokevolume_high,
                             strokevolume_arity };

  // Variable: low output error (7)
  enum errlowoutput_values { errlowoutput_true,
                             errlowoutput_false,
                             errlowoutput_arity };

  // Variable: heartrate/blood pressure (8)
  enum hrbp_values { hrbp_low,
                     hrbp_normal,
                     hrbp_high,
                     hrbp_arity };

  // Variable: heartrate/EKG (9)
  enum hrekg_values { hrekg_low,
                      hrekg_normal,
                      hrekg_high,
                      hrekg_arity };

  // Variable: error in cauterization (10)
  enum errcauter_values { errcauter_true,
                          errcauter_false,
                          errcauter_arity };

  // Variable: heart rate ??? (11)
  enum hrsat_values { hrsat_low,
                      hrsat_normal,
                      hrsat_high,
                      hrsat_arity };

  // Variable: insufficient anesthesia (12)
  enum insuffanesth_values { insuffanesth_true,
                             insuffanesth_false,
                             insuffanesth_arity  };

  // Variable: anaphylaxis (13)
  enum anaphylaxis_values { anaphylaxis_true,
                            anaphylaxis_false,
                            anaphylaxis_arity };

  // Variable: TPR (14)
  enum tpr_values { tpr_low,
                      tpr_normal,
                      tpr_high,
                      tpr_arity };

  // Variable: expelled CO2 (15)
  enum expco2_values { expco2_zero,
                       expco2_low,
                       expco2_normal,
                       expco2_high,
                       expco2_arity };

  // Variable: kinked tube (16)
  enum kinkedtube_values { kinkedtube_true,
                           kinkedtube_false,
                           kinkedtube_arity };

  // Variable: minimum volume (17)
  enum minvol_values { minvol_zero,
                       minvol_low,
                       minvol_normal,
                       minvol_high,
                       minvol_arity };

  // Variable: FiO2 (18)
  enum fio2_values { fio2_low,
                       fio2_normal,
                       fio2_arity };

  // Variable: PVSAT (19)
  enum pvsat_values { pvsat_low,
                      pvsat_normal,
                      pvsat_high,
                      pvsat_arity };

  // Variable: SAO2 (20)
  enum sao2_values { sao2_low,
                     sao2_normal,
                     sao2_high,
                     sao2_arity };

  // Variable: PAP (21)
  enum pap_values { pap_low,
                    pap_normal,
                    pap_high,
                    pap_arity };

  // Variable: pulmembolus (22)
  enum pulmembolus_values { pulmembolus_true,
                            pulmembolus_false,
                            pulmembolus_arity };

  // Variable: shunt (23)
  enum shunt_values { shunt_normal,
                      shunt_high,
                      shunt_arity };

  // Variable: intubation (24)
  enum intubation_values { intubation_normal,
                           intubation_esophageal,
                           intubation_onesided,
                           intubation_arity };

  // Variable: press (25)
  enum press_values { press_zero,
                      press_low,
                      press_normal,
                      press_high,
                      press_arity };

  // Variable: disconnect (26)
  enum disconnect_values { disconnect_true,
                           disconnect_false,
                           disconnect_arity };

  // Variable: minvolset (27)
  enum minvolset_values { minvolset_low,
                          minvolset_normal,
                          minvolset_high,
                          minvolset_arity };

  // Variable: ventilation machine (28)
  enum ventmach_values { ventmach_zero,
                         ventmach_low,
                         ventmach_normal,
                         ventmach_high,
                         ventmach_arity };

  // Variable: ventilation tube (29)
  enum venttube_values { venttube_zero,
                         venttube_low,
                         venttube_normal,
                         venttube_high,
                         venttube_arity };

  // Variable: ventilation lung (30)
  enum ventlung_values { ventlung_zero,
                         ventlung_low,
                         ventlung_normal,
                         ventlung_high,
                         ventlung_arity };

  // Variable: ventilation alv (31)
  enum ventalv_values { ventalv_zero,
                        ventalv_low,
                        ventalv_normal,
                        ventalv_high,
                        ventalv_arity };

  // Variable: arterial CO2 (32)
  enum artco2_values { artco2_low,
                       artco2_normal,
                       artco2_high,
                       artco2_arity };

  // Variable: catechol (33)
  enum catechol_values { catechol_normal,
                         catechol_high,
                         catechol_arity };

  // Variable: heart rate (34)
  enum hr_values { hr_low,
                   hr_normal,
                   hr_high,
                   hr_arity };

  // Variable: carbon monoxide (35)
  enum co_values { co_low,
                   co_normal,
                   co_high,
                   co_arity };

  // Variable: blood pressure (36)
  enum bp_values { bp_low,
                   bp_normal,
                   bp_high,
                   bp_arity };

  variable history;
  variable cvp;
  variable pcwp;
  variable hypovolemia;
  variable lvedvolume;
  variable lvfailure;
  variable strokevolume;
  variable errlowoutput;
  variable hrbp;
  variable hrekg;
  variable errcauter;
  variable hrsat;
  variable insuffanesth;
  variable anaphylaxis;
  variable tpr;
  variable expco2;
  variable kinkedtube;
  variable minvol;
  variable fio2;
  variable pvsat;
  variable sao2;
  variable pap;
  variable pulmembolus;
  variable shunt;
  variable intubation;
  variable press;
  variable disconnect;
  variable minvolset;
  variable ventmach;
  variable venttube;
  variable ventlung;
  variable ventalv;
  variable artco2;
  variable catechol;
  variable hr;
  variable co;
  variable bp;

  // Factors
  // =========================================================================
  std::vector<ptable> factors;
};


/*
    (add-factor (init-table-factor (list LVFAILURE HISTORY)
                                   '((TRUE) 0.9 0.1)
                                   '((FALSE) 0.01 0.99))
                gm)
    (add-factor (init-table-factor (list LVEDVOLUME CVP)
                                   '((LOW) 0.95 0.04 0.01)
                                   '((NORMAL) 0.04 0.95 0.01)
                                   '((HIGH) 0.01 0.29 0.7))
                gm)
    (add-factor (init-table-factor (list LVEDVOLUME PCWP)
                                   '((LOW) 0.95 0.04 0.01)
                                   '((NORMAL) 0.04 0.95 0.01)
                                   '((HIGH) 0.01 0.04 0.95))
                gm)
    (add-factor (init-table-factor (list HYPOVOLEMIA) '(() 0.2 0.8))
                gm)
    (add-factor (init-table-factor (list HYPOVOLEMIA LVFAILURE LVEDVOLUME)
                                   '((TRUE TRUE) 0.95 0.04 0.01)
                                   '((FALSE TRUE) 0.98 0.01 0.01)
                                   '((TRUE FALSE) 0.01 0.09 0.9)
                                   '((FALSE FALSE) 0.05 0.9 0.05))
                gm)
    (add-factor (init-table-factor (list LVFAILURE) '(() 0.05 0.95))
                gm)
    (add-factor (init-table-factor (list HYPOVOLEMIA LVFAILURE STROKEVOLUME)
                                   '((TRUE TRUE) 0.98 0.01 0.01)
                                   '((FALSE TRUE) 0.95 0.04 0.01)
                                   '((TRUE FALSE) 0.5 0.49 0.01)
                                   '((FALSE FALSE) 0.05 0.9 0.05))
                gm)
    (add-factor (init-table-factor (list ERRLOWOUTPUT) '(() 0.05 0.95))
                gm)
    (add-factor (init-table-factor (list ERRLOWOUTPUT HR HRBP)
                                   '((TRUE LOW) 0.98 0.01 0.01)
                                   '((FALSE LOW) 0.4 0.59 0.01)
                                   '((TRUE NORMAL) 0.3 0.4 0.3)
                                   '((FALSE NORMAL) 0.98 0.01 0.01)
                                   '((TRUE HIGH) 0.01 0.98 0.01)
                                   '((FALSE HIGH) 0.01 0.01 0.98))
                gm)
    (add-factor (init-table-factor (list ERRCAUTER HR HREKG)
                                   '((TRUE LOW) 0.33333334 0.33333334 0.33333334)
                                   '((FALSE LOW) 0.33333334 0.33333334 0.33333334)
                                   '((TRUE NORMAL) 0.33333334 0.33333334 0.33333334)
                                   '((FALSE NORMAL) 0.98 0.01 0.01)
                                   '((TRUE HIGH) 0.01 0.98 0.01)
                                   '((FALSE HIGH) 0.01 0.01 0.98))
                gm)
    (add-factor (init-table-factor (list ERRCAUTER) '(() 0.1 0.9))
                gm)
    (add-factor (init-table-factor (list ERRCAUTER HR HRSAT)
                                   '((TRUE LOW) 0.33333334 0.33333334 0.33333334)
                                   '((FALSE LOW) 0.33333334 0.33333334 0.33333334)
                                   '((TRUE NORMAL) 0.33333334 0.33333334 0.33333334)
                                   '((FALSE NORMAL) 0.98 0.01 0.01)
                                   '((TRUE HIGH) 0.01 0.98 0.01)
                                   '((FALSE HIGH) 0.01 0.01 0.98))
                gm)
    (add-factor (init-table-factor (list INSUFFANESTH) '(() 0.1 0.9))
                gm)
    (add-factor (init-table-factor (list ANAPHYLAXIS) '(() 0.01 0.99))
                gm)
    (add-factor (init-table-factor (list ANAPHYLAXIS TPR)
                                   '((TRUE) 0.98 0.01 0.01)
                                   '((FALSE) 0.3 0.4 0.3))
                gm)
    (add-factor (init-table-factor (list ARTCO2 VENTLUNG EXPCO2)
                                   '((LOW ZERO) 0.97 0.01 0.01 0.01)
                                   '((NORMAL ZERO) 0.01 0.97 0.01 0.01)
                                   '((HIGH ZERO) 0.01 0.97 0.01 0.01)
                                   '((LOW LOW) 0.01 0.97 0.01 0.01)
                                   '((NORMAL LOW) 0.97 0.01 0.01 0.01)
                                   '((HIGH LOW) 0.01 0.01 0.97 0.01)
                                   '((LOW NORMAL) 0.01 0.01 0.97 0.01)
                                   '((NORMAL NORMAL) 0.01 0.01 0.97 0.01)
                                   '((HIGH NORMAL) 0.97 0.01 0.01 0.01)
                                   '((LOW HIGH) 0.01 0.01 0.01 0.97)
                                   '((NORMAL HIGH) 0.01 0.01 0.01 0.97)
                                   '((HIGH HIGH) 0.01 0.01 0.01 0.97))
                gm)
    (add-factor (init-table-factor (list KINKEDTUBE) '(() 0.04 0.96))
                gm)
    (add-factor (init-table-factor (list INTUBATION VENTLUNG MINVOL)
                                   '((NORMAL ZERO) 0.97 0.01 0.01 0.01)
                                   '((ESOPHAGEAL ZERO) 0.01 0.97 0.01 0.01)
                                   '((ONESIDED ZERO) 0.01 0.01 0.97 0.01)
                                   '((NORMAL LOW) 0.01 0.01 0.01 0.97)
                                   '((ESOPHAGEAL LOW) 0.97 0.01 0.01 0.01)
                                   '((ONESIDED LOW) 0.6 0.38 0.01 0.01)
                                   '((NORMAL NORMAL) 0.5 0.48 0.01 0.01)
                                   '((ESOPHAGEAL NORMAL) 0.5 0.48 0.01 0.01)
                                   '((ONESIDED NORMAL) 0.97 0.01 0.01 0.01)
                                   '((NORMAL HIGH) 0.01 0.97 0.01 0.01)
                                   '((ESOPHAGEAL HIGH) 0.01 0.01 0.97 0.01)
                                   '((ONESIDED HIGH) 0.01 0.01 0.01 0.97))
                gm)
    (add-factor (init-table-factor (list FIO2) '(() 0.05 0.95))
                gm)
    (add-factor (init-table-factor (list FIO2 VENTALV PVSAT)
                                   '((LOW ZERO) 1.0 0.0 0.0)
                                   '((NORMAL ZERO) 0.99 0.01 0.0)
                                   '((LOW LOW) 0.95 0.04 0.01)
                                   '((NORMAL LOW) 0.95 0.04 0.01)
                                   '((LOW NORMAL) 1.0 0.0 0.0)
                                   '((NORMAL NORMAL) 0.95 0.04 0.01)
                                   '((LOW HIGH) 0.01 0.95 0.04)
                                   '((NORMAL HIGH) 0.01 0.01 0.98))
                gm)
    (add-factor (init-table-factor (list PVSAT SHUNT SAO2)
                                   '((LOW NORMAL) 0.98 0.01 0.01)
                                   '((NORMAL NORMAL) 0.01 0.98 0.01)
                                   '((HIGH NORMAL) 0.01 0.01 0.98)
                                   '((LOW HIGH) 0.98 0.01 0.01)
                                   '((NORMAL HIGH) 0.98 0.01 0.01)
                                   '((HIGH HIGH) 0.69 0.3 0.01))
                gm)
    (add-factor (init-table-factor (list PULMEMBOLUS PAP)
                                   '((TRUE) 0.01 0.19 0.8)
                                   '((FALSE) 0.05 0.9 0.05))
                gm)
    (add-factor (init-table-factor (list PULMEMBOLUS) '(() 0.01 0.99))
                gm)
    (add-factor (init-table-factor (list INTUBATION PULMEMBOLUS SHUNT)
                                   '((NORMAL TRUE) 0.1 0.9)
                                   '((ESOPHAGEAL TRUE) 0.1 0.9)
                                   '((ONESIDED TRUE) 0.01 0.99)
                                   '((NORMAL FALSE) 0.95 0.05)
                                   '((ESOPHAGEAL FALSE) 0.95 0.05)
                                   '((ONESIDED FALSE) 0.05 0.95))
                gm)
    (add-factor (init-table-factor (list INTUBATION) '(() 0.92 0.03 0.05))
                gm)
    (add-factor (init-table-factor (list INTUBATION KINKEDTUBE VENTTUBE PRESS)
                                   '((NORMAL TRUE ZERO) 0.97 0.01 0.01 0.01)
                                   '((ESOPHAGEAL TRUE ZERO) 0.01 0.3 0.49 0.2)
                                   '((ONESIDED TRUE ZERO) 0.01 0.01 0.08 0.9)
                                   '((NORMAL FALSE ZERO) 0.01 0.01 0.01 0.97)
                                   '((ESOPHAGEAL FALSE ZERO) 0.97 0.01 0.01 0.01)
                                   '((ONESIDED FALSE ZERO) 0.1 0.84 0.05 0.01)
                                   '((NORMAL TRUE LOW) 0.05 0.25 0.25 0.45)
                                   '((ESOPHAGEAL TRUE LOW) 0.01 0.15 0.25 0.59)
                                   '((ONESIDED TRUE LOW) 0.97 0.01 0.01 0.01)
                                   '((NORMAL FALSE LOW) 0.01 0.29 0.3 0.4)
                                   '((ESOPHAGEAL FALSE LOW) 0.01 0.01 0.08 0.9)
                                   '((ONESIDED FALSE LOW) 0.01 0.01 0.01 0.97)
                                   '((NORMAL TRUE NORMAL) 0.97 0.01 0.01 0.01)
                                   '((ESOPHAGEAL TRUE NORMAL) 0.01 0.97 0.01 0.01)
                                   '((ONESIDED TRUE NORMAL) 0.01 0.01 0.97 0.01)
                                   '((NORMAL FALSE NORMAL) 0.01 0.01 0.01 0.97)
                                   '((ESOPHAGEAL FALSE NORMAL) 0.97 0.01 0.01 0.01)
                                   '((ONESIDED FALSE NORMAL) 0.4 0.58 0.01 0.01)
                                   '((NORMAL TRUE HIGH) 0.2 0.75 0.04 0.01)
                                   '((ESOPHAGEAL TRUE HIGH) 0.2 0.7 0.09 0.01)
                                   '((ONESIDED TRUE HIGH) 0.97 0.01 0.01 0.01)
                                   '((NORMAL FALSE HIGH) 0.010000001 0.90000004 0.080000006 0.010000001)
                                   '((ESOPHAGEAL FALSE HIGH) 0.01 0.01 0.38 0.6)
                                   '((ONESIDED FALSE HIGH) 0.01 0.01 0.01 0.97))
                gm)
    (add-factor (init-table-factor (list DISCONNECT) '(() 0.1 0.9))
                gm)
    (add-factor (init-table-factor (list MINVOLSET) '(() 0.05 0.9 0.05))
                gm)
    (add-factor (init-table-factor (list MINVOLSET VENTMACH)
                                   '((LOW) 0.05 0.93 0.01 0.01)
                                   '((NORMAL) 0.05 0.01 0.93 0.01)
                                   '((HIGH) 0.05 0.01 0.01 0.93))
                gm)
    (add-factor (init-table-factor (list DISCONNECT VENTMACH VENTTUBE)
                                   '((TRUE ZERO) 0.97 0.01 0.01 0.01)
                                   '((FALSE ZERO) 0.97 0.01 0.01 0.01)
                                   '((TRUE LOW) 0.97 0.01 0.01 0.01)
                                   '((FALSE LOW) 0.97 0.01 0.01 0.01)
                                   '((TRUE NORMAL) 0.97 0.01 0.01 0.01)
                                   '((FALSE NORMAL) 0.01 0.97 0.01 0.01)
                                   '((TRUE HIGH) 0.01 0.01 0.97 0.01)
                                   '((FALSE HIGH) 0.01 0.01 0.01 0.97))
                gm)
    (add-factor (init-table-factor (list INTUBATION KINKEDTUBE VENTTUBE VENTLUNG)
                                   '((NORMAL TRUE ZERO) 0.97 0.01 0.01 0.01)
                                   '((ESOPHAGEAL TRUE ZERO) 0.95000005 0.030000001 0.010000001 0.010000001)
                                   '((ONESIDED TRUE ZERO) 0.4 0.58 0.01 0.01)
                                   '((NORMAL FALSE ZERO) 0.3 0.68 0.01 0.01)
                                   '((ESOPHAGEAL FALSE ZERO) 0.97 0.01 0.01 0.01)
                                   '((ONESIDED FALSE ZERO) 0.97 0.01 0.01 0.01)
                                   '((NORMAL TRUE LOW) 0.97 0.01 0.01 0.01)
                                   '((ESOPHAGEAL TRUE LOW) 0.97 0.01 0.01 0.01)
                                   '((ONESIDED TRUE LOW) 0.97 0.01 0.01 0.01)
                                   '((NORMAL FALSE LOW) 0.95000005 0.030000001 0.010000001 0.010000001)
                                   '((ESOPHAGEAL FALSE LOW) 0.5 0.48 0.01 0.01)
                                   '((ONESIDED FALSE LOW) 0.3 0.68 0.01 0.01)
                                   '((NORMAL TRUE NORMAL) 0.97 0.01 0.01 0.01)
                                   '((ESOPHAGEAL TRUE NORMAL) 0.01 0.97 0.01 0.01)
                                   '((ONESIDED TRUE NORMAL) 0.01 0.01 0.97 0.01)
                                   '((NORMAL FALSE NORMAL) 0.01 0.01 0.01 0.97)
                                   '((ESOPHAGEAL FALSE NORMAL) 0.97 0.01 0.01 0.01)
                                   '((ONESIDED FALSE NORMAL) 0.97 0.01 0.01 0.01)
                                   '((NORMAL TRUE HIGH) 0.97 0.01 0.01 0.01)
                                   '((ESOPHAGEAL TRUE HIGH) 0.97 0.01 0.01 0.01)
                                   '((ONESIDED TRUE HIGH) 0.97 0.01 0.01 0.01)
                                   '((NORMAL FALSE HIGH) 0.01 0.97 0.01 0.01)
                                   '((ESOPHAGEAL FALSE HIGH) 0.01 0.01 0.97 0.01)
                                   '((ONESIDED FALSE HIGH) 0.01 0.01 0.01 0.97))
                gm)
    (add-factor (init-table-factor (list INTUBATION VENTLUNG VENTALV)
                                   '((NORMAL ZERO) 0.97 0.01 0.01 0.01)
                                   '((ESOPHAGEAL ZERO) 0.01 0.97 0.01 0.01)
                                   '((ONESIDED ZERO) 0.01 0.01 0.97 0.01)
                                   '((NORMAL LOW) 0.01 0.01 0.01 0.97)
                                   '((ESOPHAGEAL LOW) 0.97 0.01 0.01 0.01)
                                   '((ONESIDED LOW) 0.01 0.97 0.01 0.01)
                                   '((NORMAL NORMAL) 0.01 0.01 0.97 0.01)
                                   '((ESOPHAGEAL NORMAL) 0.01 0.01 0.01 0.97)
                                   '((ONESIDED NORMAL) 0.97 0.01 0.01 0.01)
                                   '((NORMAL HIGH) 0.030000001 0.95000005 0.010000001 0.010000001)
                                   '((ESOPHAGEAL HIGH) 0.01 0.94 0.04 0.01)
                                   '((ONESIDED HIGH) 0.01 0.88 0.1 0.01))
                gm)
    (add-factor (init-table-factor (list VENTALV ARTCO2)
                                   '((ZERO) 0.01 0.01 0.98)
                                   '((LOW) 0.01 0.01 0.98)
                                   '((NORMAL) 0.04 0.92 0.04)
                                   '((HIGH) 0.9 0.09 0.01))
                gm)
    (add-factor (init-table-factor (list ARTCO2 INSUFFANESTH SAO2 TPR CATECHOL)
                                   '((LOW TRUE LOW LOW) 0.01 0.99)
                                   '((NORMAL TRUE LOW LOW) 0.01 0.99)
                                   '((HIGH TRUE LOW LOW) 0.01 0.99)
                                   '((LOW FALSE LOW LOW) 0.01 0.99)
                                   '((NORMAL FALSE LOW LOW) 0.01 0.99)
                                   '((HIGH FALSE LOW LOW) 0.01 0.99)
                                   '((LOW TRUE NORMAL LOW) 0.01 0.99)
                                   '((NORMAL TRUE NORMAL LOW) 0.01 0.99)
                                   '((HIGH TRUE NORMAL LOW) 0.01 0.99)
                                   '((LOW FALSE NORMAL LOW) 0.01 0.99)
                                   '((NORMAL FALSE NORMAL LOW) 0.01 0.99)
                                   '((HIGH FALSE NORMAL LOW) 0.01 0.99)
                                   '((LOW TRUE HIGH LOW) 0.01 0.99)
                                   '((NORMAL TRUE HIGH LOW) 0.01 0.99)
                                   '((HIGH TRUE HIGH LOW) 0.01 0.99)
                                   '((LOW FALSE HIGH LOW) 0.05 0.95)
                                   '((NORMAL FALSE HIGH LOW) 0.05 0.95)
                                   '((HIGH FALSE HIGH LOW) 0.01 0.99)
                                   '((LOW TRUE LOW NORMAL) 0.01 0.99)
                                   '((NORMAL TRUE LOW NORMAL) 0.01 0.99)
                                   '((HIGH TRUE LOW NORMAL) 0.01 0.99)
                                   '((LOW FALSE LOW NORMAL) 0.05 0.95)
                                   '((NORMAL FALSE LOW NORMAL) 0.05 0.95)
                                   '((HIGH FALSE LOW NORMAL) 0.01 0.99)
                                   '((LOW TRUE NORMAL NORMAL) 0.05 0.95)
                                   '((NORMAL TRUE NORMAL NORMAL) 0.05 0.95)
                                   '((HIGH TRUE NORMAL NORMAL) 0.01 0.99)
                                   '((LOW FALSE NORMAL NORMAL) 0.05 0.95)
                                   '((NORMAL FALSE NORMAL NORMAL) 0.05 0.95)
                                   '((HIGH FALSE NORMAL NORMAL) 0.01 0.99)
                                   '((LOW TRUE HIGH NORMAL) 0.05 0.95)
                                   '((NORMAL TRUE HIGH NORMAL) 0.05 0.95)
                                   '((HIGH TRUE HIGH NORMAL) 0.01 0.99)
                                   '((LOW FALSE HIGH NORMAL) 0.05 0.95)
                                   '((NORMAL FALSE HIGH NORMAL) 0.05 0.95)
                                   '((HIGH FALSE HIGH NORMAL) 0.01 0.99)
                                   '((LOW TRUE LOW HIGH) 0.7 0.3)
                                   '((NORMAL TRUE LOW HIGH) 0.7 0.3)
                                   '((HIGH TRUE LOW HIGH) 0.1 0.9)
                                   '((LOW FALSE LOW HIGH) 0.7 0.3)
                                   '((NORMAL FALSE LOW HIGH) 0.7 0.3)
                                   '((HIGH FALSE LOW HIGH) 0.1 0.9)
                                   '((LOW TRUE NORMAL HIGH) 0.7 0.3)
                                   '((NORMAL TRUE NORMAL HIGH) 0.7 0.3)
                                   '((HIGH TRUE NORMAL HIGH) 0.1 0.9)
                                   '((LOW FALSE NORMAL HIGH) 0.95 0.05)
                                   '((NORMAL FALSE NORMAL HIGH) 0.99 0.01)
                                   '((HIGH FALSE NORMAL HIGH) 0.3 0.7)
                                   '((LOW TRUE HIGH HIGH) 0.95 0.05)
                                   '((NORMAL TRUE HIGH HIGH) 0.99 0.01)
                                   '((HIGH TRUE HIGH HIGH) 0.3 0.7)
                                   '((LOW FALSE HIGH HIGH) 0.95 0.05)
                                   '((NORMAL FALSE HIGH HIGH) 0.99 0.01)
                                   '((HIGH FALSE HIGH HIGH) 0.3 0.7))
                gm)
    (add-factor (init-table-factor (list CATECHOL HR)
                                   '((NORMAL) 0.05 0.9 0.05)
                                   '((HIGH) 0.01 0.09 0.9))
                gm)
    (add-factor (init-table-factor (list HR STROKEVOLUME CO)
                                   '((LOW LOW) 0.98 0.01 0.01)
                                   '((NORMAL LOW) 0.95 0.04 0.01)
                                   '((HIGH LOW) 0.8 0.19 0.01)
                                   '((LOW NORMAL) 0.95 0.04 0.01)
                                   '((NORMAL NORMAL) 0.04 0.95 0.01)
                                   '((HIGH NORMAL) 0.01 0.04 0.95)
                                   '((LOW HIGH) 0.3 0.69 0.01)
                                   '((NORMAL HIGH) 0.01 0.3 0.69)
                                   '((HIGH HIGH) 0.01 0.01 0.98))
                gm)
    (add-factor (init-table-factor (list CO TPR BP)
                                   '((LOW LOW) 0.98 0.01 0.01)
                                   '((NORMAL LOW) 0.98 0.01 0.01)
                                   '((HIGH LOW) 0.9 0.09 0.01)
                                   '((LOW NORMAL) 0.98 0.01 0.01)
                                   '((NORMAL NORMAL) 0.1 0.85 0.05)
                                   '((HIGH NORMAL) 0.05 0.2 0.75)
                                   '((LOW HIGH) 0.3 0.6 0.1)
                                   '((NORMAL HIGH) 0.05 0.4 0.55)
                                   '((HIGH HIGH) 0.01 0.09 0.9))
                gm)
    gm))

*/

#endif
