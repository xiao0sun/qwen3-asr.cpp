#include "forced_aligner.h"

#include <iostream>
#include <string>
#include <vector>

int main() {
    qwen3_asr::ForcedAligner aligner;
    std::vector<std::string> words;

    aligner.tokenize_with_timestamps(
        "うちの中学は弁当制で持っていけない場合は50円の学校販売のパンを買う。",
        words,
        "japanese");

    if (words.size() <= 1) {
        std::cerr << "expected Japanese text to be split into multiple alignment units, got "
                  << words.size() << "\n";
        return 1;
    }
    if (words.front() != "う") {
        std::cerr << "expected first alignment unit to be う, got " << words.front() << "\n";
        return 1;
    }

    return 0;
}
