#include <iostream>

bool runParserTests();
bool runBitLengthSetTests();
bool runEvaluatorTests();
bool runAnalyzerTests();
bool runRuntimeTests();

int main() {
  bool ok = true;
  ok = runParserTests() && ok;
  ok = runBitLengthSetTests() && ok;
  ok = runEvaluatorTests() && ok;
  ok = runAnalyzerTests() && ok;
  ok = runRuntimeTests() && ok;
  if (!ok) {
    std::cerr << "unit tests failed\n";
    return 1;
  }
  std::cout << "unit tests passed\n";
  return 0;
}
