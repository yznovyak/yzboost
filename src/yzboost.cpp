#include <algorithm>
#include <cstdio>
#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <cassert>

#include "common.h"
#include "log.h"
#include "util.h"
#include "csv.h"
#include "classifier.h"

using std::cout;
using std::cerr;
using std::endl;
using std::vector;
using std::string;

using yzboost::Vector;
using yzboost::Matrix;
using yzboost::csv::read_csv;
using yzboost::util::now;
using yzboost::util::T;
using yzboost::util::seconds;
using yzboost::util::seconds_since;
using yzboost::util::extract_column;
using yzboost::util::extract_validation_split;
using yzboost::YZBoostClassifier;

// Global level and very common variables

string FLAG_action;
string FLAG_input;
string FLAG_output;
string FLAG_model;
float FLAG_validation_fraction = 0;

Matrix X_train, X_valid;
Vector y_train, y_valid;
YZBoostClassifier yzbc;

void print_usage(string argv0) {
  cout << "Usage: " << argv0 << " <action> [<options>]" << endl;
  cout << "Where <action> is one of:" << endl;
  cout << "  train" << endl;
  cout << "  predict" << endl;
  cout << endl;
  cout << "To train" << endl;
  cout << " $ " << argv0 << " train --input=data.csv --model=higgs-model.txt --tree-depth=10 --num-trees=100 --validation-fraction=0.1" << endl;
  cout << endl;
  cout << "To predict" << endl;
  cout << " $ " << argv0 << " predict --input=data.csv --model=higgs-model.txt --output=higgs-predictions.txt" << endl;
  cout << endl;
  cout << "All options:" << endl;
  cout << "  --input=somefile.csv            input file with data" << endl;
  cout << "  --output=predictions            where will the predictions go" << endl;
  cout << "  --model=higgs-model.txt         where will the model be stored or loaded from" << endl;
  cout << "  --num-trees=100                 number of trees to use" << endl;
  cout << "  --tree-depth=10                 how big those trees should be" << endl;
  cout << "  --validation-fraction=0.1       which fraction of samples to use as validation split" << endl;
  cout << "  --seed=123                      random seed" << endl;
  cout << "  --logging-enabled=0             to print to screen" << endl;
  cout << "  --silent=1                      to shut up" << endl;
  cout << "  --show-file=1                   to see where the error is coming from" << endl;
  cout << "  --discretization-bins=100       how many bins to use for discretization" << endl;
  cout << "  --discretization-samples=50000  how many samples to use to do feature binarization" << endl;
}

bool parse_arguments(int argc, char** argv) {
  if (argc < 2) {
    cerr << "Expected at least 1 command line argument" << endl;
    return false;
  }

  FLAG_action = argv[1];
  for (int i = 2; i < argc; i++) {
    string flag = argv[i];

    if (flag.find("--") != 0) {
      cerr << "Failed to parse flag #" << (i-1) << ": expected flag to start with --, but got: " << flag << endl;
      return false;
    }
    int equals_pos = flag.find("=");
    if (equals_pos == -1) {
      cerr << "Failed to parse flag #" << (i-1) << ": expected flag to have = char, but got: " << flag << endl;
      return false;
    }

    string flag_name = flag.substr(2, equals_pos-2);
    string flag_value = flag.substr(equals_pos+1);
    if (flag_name == "input") {
      FLAG_input = flag_value;
      continue;
    }
    if (flag_name == "output") {
      FLAG_output = flag_value;
      continue;
    }
    if (flag_name == "model") {
      FLAG_model = flag_value;
      continue;
    }
    if (flag_name == "tree-depth") {
      yzbc.opts.tree_depth = stoi(flag_value);
      continue;
    }
    if (flag_name == "num-trees") {
      yzbc.opts.num_trees = stoi(flag_value);
      continue;
    }
    if (flag_name == "validation-fraction") {
      FLAG_validation_fraction = stof(flag_value);
      continue;
    }
    if (flag_name == "seed") {
      yzboost::default_prng.seed(stoi(flag_value));
      continue;
    }
    if (flag_name == "logging-enabled") {
      yzboost::logging::FLAG_logging_enabled = stoi(flag_value);
      continue;
    }
    if (flag_name == "silent") {
      yzboost::logging::FLAG_logging_enabled = !stoi(flag_value);
      continue;
    }
    if (flag_name == "show-file") {
      yzboost::logging::FLAG_show_file = stoi(flag_value);
      continue;
    }
    if (flag_name == "discretization-bins") {
      yzbc.opts.discretization_max_bins = stoi(flag_value);
      continue;
    }
    if (flag_name == "discretization-samples") {
      yzbc.opts.discretization_num_subsamples = stoi(flag_value);
      continue;
    }

    cerr << "Failed to parse flag #" << (i-1) << ": unrecognized flag " << flag_name << endl;
    return false;
  }

  return true;
}

int train() {
  if (FLAG_input.empty()) { LOG <<  "--input is required"; return 1; }
  if (FLAG_model.empty()) { LOG <<  "--model is required"; return 1; }

  T t0 = now();
  LOG << "Reading input ...";
  if (!read_csv(FLAG_input, X_train)) {
    LOG << "Failed to read data";
    return 1;
  }
  if (X_train.size() == 0) {
    LOG << "Load an empty matrix";
    return 1;
  }
  LOG << "Loaded " << X_train.size() << " x " << X_train[0].size() << " float matrix in " << seconds_since(t0) << " sec";

  if (FLAG_validation_fraction > 0) {
    LOG << "Splitting @ " << FLAG_validation_fraction;
    X_valid = extract_validation_split(X_train, FLAG_validation_fraction);
    y_valid = extract_column(X_valid, 0);
    yzbc.opts.valid_X = &X_valid;
    yzbc.opts.valid_y = &y_valid;
  }
  y_train = extract_column(X_train, 0);

  LOG << "X_train of shape " << X_train.size() << " x " << X_train[0].size() << "     y_train of shape " << y_train.size();
  if (!X_valid.empty())
    LOG << "X_valid of shape " << X_valid.size() << " x " << X_valid[0].size() << "     y_valid of shape " << y_valid.size();

  t0 = now();
  yzbc.fit(X_train, y_train);
  LOG << "Fit in " << seconds_since(t0);

  {
    std::ofstream f(FLAG_model);
    yzbc.save(f);
  }

  YZBoostClassifier yzbc2;
  {
    std::ifstream f(FLAG_model);
    yzbc2.load(f);
  }

  assert(yzbc.get_bias() == yzbc2.get_bias());
  for (size_t i = 0; i < yzbc.get_trees().size(); i++) {
    yzboost::Tree t1 = yzbc.get_trees()[i], t2 = yzbc2.get_trees()[i];
    assert(t1.depth() == t2.depth());
    for (int j = 0; j < t1.depth(); j++) {
      assert(t1.splits[j].feature == t2.splits[j].feature);
      assert(t1.splits[j].threshold == t2.splits[j].threshold);
    }
  }

  return 0;
}

int predict() {
  if (FLAG_input.empty())  { LOG <<  "--input is required"; return 1; }
  if (FLAG_model.empty())  { LOG <<  "--model is required"; return 1; }
  if (FLAG_output.empty()) { LOG << "--output is required"; return 1; }

  T t0 = now();
  LOG << "Loading ...";
  if (!read_csv(FLAG_input, X_train)) {
    LOG << "Failed to read data";
    return 1;
  }
  if (X_train.size() == 0) {
    LOG << "Load an empty matrix";
    return 1;
  }
  LOG << "Loaded " << X_train.size() << " x " << X_train[0].size() << " float matrix in " << seconds_since(t0) << " sec";
  y_train = extract_column(X_train, 0);

  std::ifstream f(FLAG_model);
  yzbc.load(f);

  LOG << "Loaded classifier with " << yzbc.get_trees().size() << " trees";
  if (X_train[0].size() != yzbc.get_num_features()) {
    LOG << "Loaded classifier that works with " << yzbc.get_num_features()
        << " features, but given matrix with " << X_train[0].size() << " features";
    return 1;
  }

  Vector prob = yzbc.predict_proba(X_train);

  std::ofstream f_out(FLAG_output);
  for (size_t i = 0; i < prob.size(); i++)
    f_out << prob[i] << "\n";

  return 0;
}

int main(int argc, char** argv) {
  if (!parse_arguments(argc, argv)) {
    print_usage(argv[0]);
    return 1;
  }

  if (FLAG_action == "train")   return train();
  if (FLAG_action == "predict") return predict();

  cerr << "Don't know how to do '" << FLAG_action << "'" << endl;
  print_usage(argv[0]);

  return 1;
}
