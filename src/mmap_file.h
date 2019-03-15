#pragma once

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <stdexcept>

#include <atomic>
#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/finder.hpp>
#include <boost/algorithm/string/predicate.hpp>
#include <boost/algorithm/string/split.hpp>
#include <boost/tokenizer.hpp>
#include <boost/utility/string_ref.hpp>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <queue>
#include <set>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <glog/logging.h>

class MmapReadableFile {
 public:
  static size_t GetFileSize(const std::string &filename) {
    struct ::stat file_stat;
    if (::stat(filename.c_str(), &file_stat) != 0) {
      throw std::runtime_error("Open file failed:" + filename);
    }
    return file_stat.st_size;
  }

  std::unique_ptr<MmapReadableFile> static NewRandomAccessFile(
      const std::string &filename) {
    size_t file_size = GetFileSize(filename);
    int fd = ::open(filename.c_str(), O_RDONLY);
    if (fd < 0) {
      throw std::runtime_error("Open file failed:" + filename);
    }
    void *mmap_base =
        ::mmap(/*addr=*/nullptr, file_size, PROT_READ, MAP_SHARED, fd, 0);
    DLOG(INFO) << "Start address of map file:" << std::hex << mmap_base
              << ", end address:"
              << (void *)(reinterpret_cast<char *>(mmap_base) + file_size);
    if (mmap_base != MAP_FAILED) {
      return std::make_unique<MmapReadableFile>(
          filename, reinterpret_cast<char *>(mmap_base), file_size);
    }
    return {};
  }

  // mmap_base[0, length-1] points to the memory-mapped contents of the file. It
  // must be the result of a successful call to mmap(). This instances takes
  // over the ownership of the region.
  MmapReadableFile(std::string filename, char *mmap_base, size_t length)
      : mmap_base_(mmap_base),
        length_(length),
        filename_(std::move(filename)) {}

  ~MmapReadableFile() { ::munmap(static_cast<void *>(mmap_base_), length_); }

  boost::string_ref FullText() { return {mmap_base_, length_}; }

  boost::string_ref Read(uint64_t offset, size_t n) {
    if (offset + n > length_) {
      return {};
    }

    return {mmap_base_ + offset, n};
  }

  std::vector<boost::string_ref> SplitBy(char sep = '\n') {
    using iterator_vec =
        std::vector<boost::iterator_range<std::string::const_iterator>>;
    iterator_vec rangeVector;
    auto fullText = FullText();
    boost::iter_split(rangeVector, fullText,
                      boost::token_finder([sep](char c) { return c == sep; }));

    std::vector<boost::string_ref> refs;
    std::transform(rangeVector.begin(), rangeVector.end(),
                   std::back_inserter(refs), [](const auto &iteratorRange) {
                     return boost::string_ref(&*iteratorRange.begin(),
                                              iteratorRange.size());
                   });

    return refs;
  }

 private:
  char *const mmap_base_;
  const size_t length_;
  const std::string filename_;
};
