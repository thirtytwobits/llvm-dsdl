#include "llvmdsdl/Semantics/BitLengthSet.h"

#include <algorithm>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace llvmdsdl {

struct BitLengthSet::Node final {
  enum class Kind {
    Leaf,
    Add,
    Union,
    Pad,
    Repeat,
    RepeatRange,
  } kind{Kind::Leaf};

  std::set<std::int64_t> values;
  std::shared_ptr<const Node> lhs;
  std::shared_ptr<const Node> rhs;
  std::int64_t param{0};

  [[nodiscard]] std::int64_t min() const {
    switch (kind) {
    case Kind::Leaf:
      return values.empty() ? 0 : *values.begin();
    case Kind::Add:
      return lhs->min() + rhs->min();
    case Kind::Union:
      return std::min(lhs->min(), rhs->min());
    case Kind::Pad: {
      const auto v = lhs->min();
      const auto a = std::max<std::int64_t>(1, param);
      const auto rem = v % a;
      return rem == 0 ? v : v + (a - rem);
    }
    case Kind::Repeat:
      return lhs->min() * std::max<std::int64_t>(0, param);
    case Kind::RepeatRange:
      return 0;
    }
    return 0;
  }

  [[nodiscard]] std::int64_t max() const {
    switch (kind) {
    case Kind::Leaf:
      return values.empty() ? 0 : *values.rbegin();
    case Kind::Add:
      return lhs->max() + rhs->max();
    case Kind::Union:
      return std::max(lhs->max(), rhs->max());
    case Kind::Pad: {
      const auto v = lhs->max();
      const auto a = std::max<std::int64_t>(1, param);
      const auto rem = v % a;
      return rem == 0 ? v : v + (a - rem);
    }
    case Kind::Repeat:
      return lhs->max() * std::max<std::int64_t>(0, param);
    case Kind::RepeatRange:
      return lhs->max() * std::max<std::int64_t>(0, param);
    }
    return 0;
  }

  [[nodiscard]] std::set<std::int64_t> expand(std::size_t limit) const {
    switch (kind) {
    case Kind::Leaf:
      return values;
    case Kind::Add: {
      std::set<std::int64_t> out;
      const auto l = lhs->expand(limit);
      const auto r = rhs->expand(limit);
      for (const auto lv : l) {
        for (const auto rv : r) {
          out.insert(lv + rv);
          if (out.size() >= limit) {
            return out;
          }
        }
      }
      return out;
    }
    case Kind::Union: {
      auto out = lhs->expand(limit);
      const auto r = rhs->expand(limit);
      out.insert(r.begin(), r.end());
      while (out.size() > limit) {
        out.erase(std::prev(out.end()));
      }
      return out;
    }
    case Kind::Pad: {
      std::set<std::int64_t> out;
      const auto l = lhs->expand(limit);
      const auto a = std::max<std::int64_t>(1, param);
      for (auto v : l) {
        const auto rem = v % a;
        if (rem != 0) {
          v += (a - rem);
        }
        out.insert(v);
        if (out.size() >= limit) {
          return out;
        }
      }
      return out;
    }
    case Kind::Repeat: {
      if (param <= 0) {
        return {0};
      }
      auto acc = std::set<std::int64_t>{0};
      const auto item = lhs->expand(limit);
      for (std::int64_t i = 0; i < param; ++i) {
        std::set<std::int64_t> next;
        for (const auto a : acc) {
          for (const auto b : item) {
            next.insert(a + b);
            if (next.size() >= limit) {
              break;
            }
          }
          if (next.size() >= limit) {
            break;
          }
        }
        acc = std::move(next);
      }
      return acc;
    }
    case Kind::RepeatRange: {
      std::set<std::int64_t> out{0};
      auto acc = std::set<std::int64_t>{0};
      const auto item = lhs->expand(limit);
      const auto maxCount = std::max<std::int64_t>(0, param);
      for (std::int64_t i = 1; i <= maxCount; ++i) {
        std::set<std::int64_t> next;
        for (const auto a : acc) {
          for (const auto b : item) {
            next.insert(a + b);
            if (next.size() >= limit) {
              break;
            }
          }
          if (next.size() >= limit) {
            break;
          }
        }
        out.insert(next.begin(), next.end());
        while (out.size() > limit) {
          out.erase(std::prev(out.end()));
        }
        acc = std::move(next);
      }
      return out;
    }
    }
    return {0};
  }

  [[nodiscard]] std::string str() const {
    std::ostringstream out;
    switch (kind) {
    case Kind::Leaf: {
      out << '{';
      bool first = true;
      for (auto v : values) {
        if (!first) {
          out << ',';
        }
        out << v;
        first = false;
      }
      out << '}';
      break;
    }
    case Kind::Add:
      out << "concat(" << lhs->str() << "," << rhs->str() << ")";
      break;
    case Kind::Union:
      out << "union(" << lhs->str() << "," << rhs->str() << ")";
      break;
    case Kind::Pad:
      out << "pad(" << lhs->str() << "," << param << ")";
      break;
    case Kind::Repeat:
      out << "repeat(" << lhs->str() << "," << param << ")";
      break;
    case Kind::RepeatRange:
      out << "repeat_range(" << lhs->str() << "," << param << ")";
      break;
    }
    return out.str();
  }
};

BitLengthSet::BitLengthSet() : root_(std::make_shared<Node>()) {
  auto leaf = std::make_shared<Node>();
  leaf->kind = Node::Kind::Leaf;
  leaf->values = {0};
  root_ = leaf;
}

BitLengthSet::BitLengthSet(std::int64_t value)
    : BitLengthSet(std::set<std::int64_t>{value}) {}

BitLengthSet::BitLengthSet(std::set<std::int64_t> values) {
  auto leaf = std::make_shared<Node>();
  leaf->kind = Node::Kind::Leaf;
  leaf->values = std::move(values);
  if (leaf->values.empty()) {
    leaf->values.insert(0);
  }
  root_ = leaf;
}

BitLengthSet::BitLengthSet(std::shared_ptr<const Node> root)
    : root_(std::move(root)) {}

std::int64_t BitLengthSet::min() const { return root_->min(); }

std::int64_t BitLengthSet::max() const { return root_->max(); }

bool BitLengthSet::fixed() const { return min() == max(); }

BitLengthSet BitLengthSet::padToAlignment(std::int64_t alignment) const {
  auto node = std::make_shared<Node>();
  node->kind = Node::Kind::Pad;
  node->lhs = root_;
  node->param = std::max<std::int64_t>(1, alignment);
  return BitLengthSet(node);
}

BitLengthSet BitLengthSet::repeat(std::int64_t count) const {
  auto node = std::make_shared<Node>();
  node->kind = Node::Kind::Repeat;
  node->lhs = root_;
  node->param = std::max<std::int64_t>(0, count);
  return BitLengthSet(node);
}

BitLengthSet BitLengthSet::repeatRange(std::int64_t countMax) const {
  auto node = std::make_shared<Node>();
  node->kind = Node::Kind::RepeatRange;
  node->lhs = root_;
  node->param = std::max<std::int64_t>(0, countMax);
  return BitLengthSet(node);
}

std::set<std::int64_t> BitLengthSet::modulo(std::int64_t divisor) const {
  if (divisor <= 0) {
    return {0};
  }
  std::set<std::int64_t> out;
  const auto expanded = expand();
  for (const auto v : expanded) {
    out.insert(v % divisor);
  }
  return out;
}

std::set<std::int64_t> BitLengthSet::expand(std::size_t limit) const {
  return root_->expand(limit);
}

std::string BitLengthSet::str() const { return root_->str(); }

BitLengthSet operator+(const BitLengthSet &lhs, const BitLengthSet &rhs) {
  auto node = std::make_shared<BitLengthSet::Node>();
  node->kind = BitLengthSet::Node::Kind::Add;
  node->lhs = lhs.root_;
  node->rhs = rhs.root_;
  return BitLengthSet(node);
}

BitLengthSet operator|(const BitLengthSet &lhs, const BitLengthSet &rhs) {
  auto node = std::make_shared<BitLengthSet::Node>();
  node->kind = BitLengthSet::Node::Kind::Union;
  node->lhs = lhs.root_;
  node->rhs = rhs.root_;
  return BitLengthSet(node);
}

} // namespace llvmdsdl
