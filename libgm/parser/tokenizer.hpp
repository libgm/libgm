#ifndef LIBGM_TOKENIZER_HPP
#define LIBGM_TOKENIZER_HPP

#include <libgm/global.hpp>

#include <string>

namespace libgm {

  /**
   * A simple tokenizer for splitting according to a delimiter string.
   */
  class tokenizer {
  private:
    const std::string s_, delims_;
    size_t position_;

  public:
    tokenizer(const std::string& s, const std::string& delims = " \t")
      : s_(s), delims_(delims) {
      position_ = s_.find_first_not_of(delims);
    };

    bool has_token() {
      assert((position_ < s_.length()) || (position_ == std::string::npos));
      return position_ < s_.length();
    }

    std::string next_token() {
      assert(has_token());
      //assert that the position_ points to a non-delimiter
      assert(delims_.find_first_of(s_[position_]) == std::string::npos);

      size_t token_end_pos = s_.find_first_of(delims_, position_);
      std::string result;
      if (token_end_pos == std::string::npos) {
        //no delimiters in the remainder, return the whole suffix
        result = s_.substr(position_);
        position_ = std::string::npos;
      } else {
        result = s_.substr(position_, token_end_pos - position_);
        position_ = s_.find_first_not_of(delims_, token_end_pos);
      }

      return result;
    }

  }; // class tokenizer

} // namespace libgm

#endif
