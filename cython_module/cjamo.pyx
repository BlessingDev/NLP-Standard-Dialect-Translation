# distutils: language = c++

from libcpp.string cimport string
from libcpp.vector cimport vector

cdef extern from "cpp_source/jamo.cpp":
    cdef string JamoToHangeul(string, string, string, vector[string])

def jamo_to_hangeul(str chosung, str jungsung, str jongsung=' '):
    cdef string cho_cstr
    cdef string jung_cstr
    cdef string jong_cstr
    cdef string result

    cho_cstr = chosung.encode('cp949')
    jung_cstr = jungsung.encode('cp949')
    jong_cstr = jongsung.encode('cp949')

    result = JamoToHangeul(cho_cstr, jung_cstr, jong_cstr, ["ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ".encode('cp949'), "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ".encode('cp949'), "ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ".encode('cp949')])

    return result.decode('cp949')