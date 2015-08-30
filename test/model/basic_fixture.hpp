#ifndef LIBGM_TEST_MODEL_BASIC_FIXTURE_HPP
#define LIBGM_TEST_MODEL_BASIC_FIXTURE_HPP

#include <libgm/argument/universe.hpp>
#include <libgm/argument/var.hpp>
#include <libgm/factor/probability_table.hpp>

#include <vector>

using namespace libgm;

typedef probability_table<var> ptable;

struct basic_fixture {

  typedef std::vector<std::string> sv;

  basic_fixture() {
    history      = var::discrete(u, "history", sv{"true", "false"});
    cvp          = var::discrete(u, "cvp", sv{"low", "normal", "high"});
    pcwp         = var::discrete(u, "pcwp", sv{"low", "normal", "high"});
    hypovolemia  = var::discrete(u, "hypovolemia", sv{"true", "false"});
    lvedvolume   = var::discrete(u, "lvedvolume", sv{"low", "normal", "high"});
    lvfailure    = var::discrete(u, "lvfailure", sv{"true", "false"});
    strokevolume = var::discrete(u, "strokevolume", sv{"low", "normal", "high"});
    errlowoutput = var::discrete(u, "errlowoutput", sv{"true", "false"});
    hrbp         = var::discrete(u, "hrbp", sv{"low", "normal", "high"});
    hrekg        = var::discrete(u, "hrekg", sv{"low", "normal", "high"});
    errcauter    = var::discrete(u, "errcauter", sv{"true", "false"});
    hrsat        = var::discrete(u, "hrsat", sv{"low", "normal", "high"});
    insuffanesth = var::discrete(u, "insuffanesth", sv{"true", "false"});
    anaphylaxis  = var::discrete(u, "anaphylaxis", sv{"true", "false"});
    tpr          = var::discrete(u, "tpr", sv{"low", "tpr", "normal", "high"});
    expco2       = var::discrete(u, "expco2", sv{"zero", "low", "normal", "high"});
    kinkedtube   = var::discrete(u, "kinkedtube", sv{"true", "false"});
    minvol       = var::discrete(u, "minvol", sv{"zero", "low", "normal", "high"});
    fio2         = var::discrete(u, "fio2", sv{"low", "normal"});
    pvsat        = var::discrete(u, "pvsat", sv{"low", "normal" "high"});
    sao2         = var::discrete(u, "sao2", sv{"low", "normal", "high"});
    pap          = var::discrete(u, "pap", sv{"pap_low", "normal", "high"});
    pulmembolus  = var::discrete(u, "pulmembolus", sv{"true", "false"});
    shunt        = var::discrete(u, "shunt", sv{"normal", "high"});
    intubation   = var::discrete(u, "intubation", sv{"normal", "esophageal", "onesided"});
    press        = var::discrete(u, "press", sv{"zero", "low", "normal", "high"});
    disconnect   = var::discrete(u, "disconnect", sv{"true", "false"});
    minvolset    = var::discrete(u, "minvolset", sv{"low", "normal", "high"});
    ventmach     = var::discrete(u, "ventmach", sv{"zero", "low", "normal", "high"});
    venttube     = var::discrete(u, "venttube", sv{"zero", "low", "normal", "high"});
    ventlung     = var::discrete(u, "ventlung", sv{"zero", "low", "normal", "high"});
    ventalv      = var::discrete(u, "ventalv", sv{"zero", "low", "normal", "high"});
    artco2       = var::discrete(u, "artco2", sv{"low", "normal", "high"});
    catechol     = var::discrete(u, "catechol", sv{"normal", "high"});
    hr           = var::discrete(u, "hr", sv{"low", "normal", "high"});
    co           = var::discrete(u, "co", sv{"low", "normal", "high"});
    bp           = var::discrete(u, "bp", sv{"low", "normal", "high"});

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

  universe u;

  var history;        // history
  var cvp;            // CVP
  var pcwp;           // PCWP
  var hypovolemia;    // hypovolemia
  var lvedvolume;     // LVED volume
  var lvfailure;      // liver failure
  var strokevolume;   // stroke volume
  var errlowoutput;   // low output error
  var hrbp;           // heartrate/blood pressure
  var hrekg;          // heartrate/EKG
  var errcauter;      // error in cauterization
  var hrsat;          // heart rate ???
  var insuffanesth;   // insufficient anesthesia
  var anaphylaxis;    // anaphylaxsi
  var tpr;            // TPR
  var expco2;         // expelled CO2
  var kinkedtube;     // kinked tube
  var minvol;         // minimum volume
  var fio2;           // FiO2
  var pvsat;          // PVSAT
  var sao2;           // SAO2
  var pap;            // PAP
  var pulmembolus;    // pulmembolus
  var shunt;          // shunt
  var intubation;     // intubation
  var press;          // press
  var disconnect;     // disconnect
  var minvolset;      // minvolset
  var ventmach;       // ventilation machine
  var venttube;       // ventilation tube
  var ventlung;       // ventilation lung
  var ventalv;        // ventilation alv
  var artco2;         // arterial CO2
  var catechol;       // catechol
  var hr;             // heart rate
  var co;             // carbon monoxide
  var bp;             // blood pressure

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
