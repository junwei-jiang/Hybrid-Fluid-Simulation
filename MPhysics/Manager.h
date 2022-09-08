#pragma once

#define Manager(T)							\
	private:								\
		T() { };							\
		~T(){ };							\
											\
		T(const T&);						\
		T& operator=(const T&);				\
											\
	public:									\
		static T& getInstance()				\
		{									\
			static T instance;				\
			return instance;				\
		}