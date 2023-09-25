#include <string>
#include <iostream>
#include <clocale>
#include <vector>

typedef unsigned short ushort;

std::string JamoToHangeul(std::string chosung, std::string jungsung, std::string jongsung, std::vector<std::string> tbls)
{
    int ChoSungPos, JungSungPos, JongSungPos;
    int nUniCode;
    std::string m_ChoSungTbl = tbls[0];

    std::string m_JungSungTbl = tbls[1];

    std::string m_JongSungTbl = tbls[2];

    ushort m_UniCodeHangulBase = 0xAC00;

    ushort m_UniCodeHangulLast = 0xD79F;

    std::cout << chosung << jungsung << jongsung << std::endl;
    //std::cout << m_ChoSungTbl << std::endl;

    ChoSungPos = m_ChoSungTbl.find(chosung) / 2;

    JungSungPos = m_JungSungTbl.find(jungsung) / 2;

    if(jongsung == " ")
    {
        JongSungPos = 0;
    }
    else
    {
        JongSungPos = m_JongSungTbl.find(jongsung) / 2 + 1;
    }

    std::cout << ChoSungPos << " " << JungSungPos << " " << JongSungPos << std::endl;

    nUniCode = m_UniCodeHangulBase + (ChoSungPos * 21 + JungSungPos) * 28 + JongSungPos;

    std::cout << "Unicode: " << nUniCode << std::endl;

    wchar_t a[15];

    swprintf(a, sizeof(a) / sizeof(wchar_t), L"%c", nUniCode);

    std::cout << a[0] << " " << a[1] << std::endl;

    char b[15];

    sprintf_s(b, 15, "%ls", a);

    std::string result(b);

    std::cout << "result: " << b << std::endl;

    return result;
}