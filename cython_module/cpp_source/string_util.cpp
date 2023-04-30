#include <iostream>
#include <vector>
#include <algorithm>
#include <sstream>
#include <iterator>
 
std::string join(std::vector<std::string> strings, std::string delim)
{
    std::stringstream ss;
    std::copy(strings.begin(), strings.end(),
        std::ostream_iterator<std::string>(ss, delim.c_str()));
    return ss.str();
}

std::string replace_all(std::string s, std::string find_str, std::string rep)
{
    auto str_pos = s.find(find_str);
    while(str_pos != std::string::npos)
    {
        s.replace(str_pos, find_str.length(), rep);
        str_pos = s.find(find_str);
    }

    return s;
}