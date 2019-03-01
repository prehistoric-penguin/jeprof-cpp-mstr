#include <iostream>
#include <string>
#include <csignal>
#include <vector>
#include <fstream>
#include <gflags/gflags.h>
#include <boost/algorithm/string/predicate.hpp>

DEFINE_string(fileName, "", "Name of execuable file");
DEFINE_string(profileName, "", "Name of profile name");

const std::string kPprofVersion = "2.0";
const std::string kObjdump = "objdump";
const std::string kNm = "nm";
const std::string kAddr2Line = "addr2line";
const std::string kCppFilt = "c++filt";
const std::string kDot = "dot";

const std::string kHeapPage = "/pprof/heap";
const std::string kProfilePage = "/pprof/prifile";
const std::string kGrowthPage = "/pprof/growth";
const std::string kContentionPage = "/pprof/contention";
const std::string kWallPage = "/pprof/wall(?:\\?.*)?";		// FIXME
const std::string KFilteredProfilePage = "/pprof/filteredprofile(?:\\?.*)?";
const std::string kSymbolPage = "/pprof/symbol";
const std::string kProgramNamePage = "/pprof/cmdline";
const std::vector<std::string> kProfiles = {
	kHeapPage,
	kProfilePage,
	kGrowthPage,
	kWallPage,
	KFilteredProfilePage,
	kSymbolPage,
	kProgramNamePage
};

const std::string kUnknownBinary = "(unknown)";
const size_t kAddressLength = 16;
const std::string kDevNull = "/dev/null";

const std::string kSepSymbol = "_fini";
const std::string kTmpFileSym = "/tmp/jeprof$$.sym";		// FIXME
const std::string kTmpFilePs = "/tmp/jeprof$$";

void SignalIntHandler(int signal) {
	std::cerr << "Signal int hit" << std::endl;				// FIXME
}

std::string ReadProfileHeader(std::ifstream& ifs) {
	const std::string profile = FLAGS_profileName;
	// Skiped check for non text charracter header, FIXME
	std::string line;
	while (std::getline(ifs, line)) {
		// FIXME replace with regex
		if (boost::starts_with(line, "warn")) {
			std::cerr << "WARNING:" << line << std::endl;
		} else if (boost::starts_with(line, "%")) {
			std::cerr << "Ignoring unknown command from profile header:"
				<< line << std::endl;
		} else {
			return line;
		}
	}
}

bool IsSymbolizedProfileFile() {
	std::ifstream ifs(FLAGS_fileName, std::ios::binary);
	auto firstLine = ReadProfileHeader(ifs);

	if (firstLine.empty()) return false;

	return true;
	//const std::string symbolPage = "m,[^/]+$,";
}

void Init() {
}

int main(int argc, char *argv[]) { 
	gflags::ParseCommandLineFlags(&argc, &argv, true);
	std::signal(SIGINT, SignalIntHandler);

	Init();
	// std::map<> sourceCache;
	//std::string profileName = "a.out";
	//std::string fileName = "a.out";
	
	
	return 0; 
}
